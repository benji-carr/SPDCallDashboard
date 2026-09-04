from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping

from forecasting.notifications import (
    SmtpEmailTransport,
    SmtpNotificationConfig,
    collect_secret_values,
    redact_secrets,
)
from forecasting.paths import FORECASTS_DIR, MONITORING_DIR, OPERATIONS_DIR, TARGET_PANEL_5Y_PATH
from forecasting.production.orchestration import run_daily_pipeline
from forecasting.production.status import format_production_status, load_production_status


LOGGER = logging.getLogger(__name__)


PipelineRunner = Callable[..., dict[str, Any]]
StatusLoader = Callable[..., dict[str, Any]]
StatusFormatter = Callable[..., str]


def _failure_files(operations_root: str | Path) -> set[Path]:
    base = Path(operations_root) / "spd_neighborhood_xgboost" / "v1" / "failures"
    return set(base.glob("execution_id=*.json")) if base.is_dir() else set()


def _read_failure_context(new_failure_files: set[Path]) -> dict[str, Any]:
    if not new_failure_files:
        return {}
    latest = max(new_failure_files, key=lambda path: (path.stat().st_mtime, str(path)))
    payload = json.loads(latest.read_text(encoding="utf-8"))
    phase_statuses = payload.get("phase_statuses", [])
    failed_phase = None
    for phase in reversed(phase_statuses):
        if phase.get("status") == "failed":
            failed_phase = phase.get("phase")
            break
    return {
        "execution_id": payload.get("execution_id"),
        "failed_phase": failed_phase,
        "manifest": payload,
    }


def _status_text(
    artifact_dir: str | Path,
    *,
    forecasts_root: str | Path,
    monitoring_root: str | Path,
    operations_root: str | Path,
    status_loader: StatusLoader,
    status_formatter: StatusFormatter,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    try:
        status = status_loader(
            artifact_dir,
            forecasts_root=forecasts_root,
            monitoring_root=monitoring_root,
            operations_root=operations_root,
        )
        return status_formatter(status), status, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def _success_subject(status: dict[str, Any]) -> str:
    artifact_id = status["artifact"]["run_id"]
    target_date = (status.get("forecast") or {}).get("target_date") or "unknown-target"
    return f"SUCCESS {artifact_id} {target_date}"


def _failure_subject(artifact_dir: str | Path, failure_context: dict[str, Any]) -> str:
    execution_id = failure_context.get("execution_id") or "unknown-execution"
    return f"FAILURE {Path(artifact_dir).name} {execution_id}"


def _success_body(result: dict[str, Any], status: dict[str, Any], status_text: str) -> str:
    forecast = status.get("forecast") or {}
    pipeline = status.get("pipeline") or {}
    artifact = status.get("artifact") or {}
    data = status.get("data") or {}
    return "\n".join(
        [
            "SPD neighborhood XGBoost production run completed successfully.",
            "",
            f"Overall success: yes",
            f"Execution ID: {result['manifest'].get('execution_id', 'N/A')}",
            f"Logical run ID: {pipeline.get('logical_run_id', 'N/A')}",
            f"Artifact ID: {artifact.get('run_id', 'N/A')}",
            f"Artifact version: {artifact.get('model_version', 'N/A')}",
            f"Latest actual date: {data.get('latest_actual', 'N/A')}",
            f"Source age: {data.get('source_age_days', 'N/A')} days",
            f"Latest forecast target date: {forecast.get('target_date', 'N/A')}",
            f"Forecast ID: {forecast.get('forecast_id', 'N/A')}",
            f"Production run idempotent: {'yes' if result.get('idempotent') else 'no'}",
            "",
            "Full status report:",
            status_text,
        ]
    )


def _failure_body(
    artifact_dir: str | Path,
    exc: Exception,
    failure_context: dict[str, Any],
    status_text: str | None,
    status_error: str | None,
) -> str:
    artifact_id = Path(artifact_dir).name
    lines = [
        "SPD neighborhood XGBoost production run failed.",
        "",
        "Overall success: no",
        f"Artifact ID: {artifact_id}",
        f"Execution ID: {failure_context.get('execution_id', 'N/A')}",
        f"Failed phase: {failure_context.get('failed_phase', 'N/A')}",
        f"Exception class: {type(exc).__name__}",
        f"Exception message: {exc}",
        "",
    ]
    if status_text is None:
        lines.extend(
            [
                "Full status report:",
                f"Status could not be generated: {status_error or 'unknown error'}",
            ]
        )
    else:
        lines.extend(["Full status report:", status_text])
    return "\n".join(lines)


def _transport_from_env(
    env: Mapping[str, str] | None = None,
    *,
    transport_factory: Callable[[SmtpNotificationConfig], Any] | None = None,
) -> tuple[Any | None, tuple[str, ...]]:
    config = SmtpNotificationConfig.from_env(env)
    secrets = collect_secret_values(env, config=config)
    if config is None:
        return None, secrets
    factory = transport_factory or SmtpEmailTransport
    return factory(config), secrets


def _send_email_or_log(
    *,
    subject: str,
    body: str,
    env: Mapping[str, str] | None,
    transport_factory: Callable[[SmtpNotificationConfig], Any] | None,
    logger: logging.Logger,
) -> None:
    try:
        transport, secrets = _transport_from_env(env, transport_factory=transport_factory)
        if transport is None:
            return
        transport.send_email(subject=subject, body=body)
    except Exception as exc:
        secrets = collect_secret_values(env)
        logger.error(
            "SPD XGBoost notification delivery failed: %s",
            redact_secrets(f"{type(exc).__name__}: {exc}", secrets),
        )


def run_daily_pipeline_with_notifications(
    *,
    artifact_dir: str | Path,
    target_panel_path: str | Path = TARGET_PANEL_5Y_PATH,
    forecasts_root: str | Path = FORECASTS_DIR,
    monitoring_root: str | Path = MONITORING_DIR,
    operations_root: str | Path = OPERATIONS_DIR,
    max_source_age_days: int | None = None,
    skip_source_refresh: bool = False,
    env: Mapping[str, str] | None = None,
    logger: logging.Logger | None = None,
    pipeline_runner: PipelineRunner | None = None,
    status_loader: StatusLoader = load_production_status,
    status_formatter: StatusFormatter = format_production_status,
    transport_factory: Callable[[SmtpNotificationConfig], Any] | None = None,
    refresh_function=None,
) -> dict[str, Any]:
    logger = logger or LOGGER
    pipeline_runner = pipeline_runner or run_daily_pipeline
    before_failures = _failure_files(operations_root)
    pipeline_kwargs = {
        "artifact_dir": artifact_dir,
        "target_panel_path": target_panel_path,
        "forecasts_root": forecasts_root,
        "monitoring_root": monitoring_root,
        "operations_root": operations_root,
        "max_source_age_days": max_source_age_days,
        "skip_source_refresh": skip_source_refresh,
    }
    if refresh_function is not None:
        pipeline_kwargs["refresh_function"] = refresh_function
    try:
        result = pipeline_runner(**pipeline_kwargs)
    except Exception as exc:
        after_failures = _failure_files(operations_root)
        failure_context = _read_failure_context(after_failures - before_failures)
        status_text, _, status_error = _status_text(
            artifact_dir,
            forecasts_root=forecasts_root,
            monitoring_root=monitoring_root,
            operations_root=operations_root,
            status_loader=status_loader,
            status_formatter=status_formatter,
        )
        _send_email_or_log(
            subject=_failure_subject(artifact_dir, failure_context),
            body=_failure_body(artifact_dir, exc, failure_context, status_text, status_error),
            env=env,
            transport_factory=transport_factory,
            logger=logger,
        )
        raise
    status_text, status, status_error = _status_text(
        artifact_dir,
        forecasts_root=forecasts_root,
        monitoring_root=monitoring_root,
        operations_root=operations_root,
        status_loader=status_loader,
        status_formatter=status_formatter,
    )
    if status_text is None or status is None:
        logger.warning(
            "SPD XGBoost status report could not be generated after successful pipeline completion: %s",
            status_error,
        )
        status = {
            "artifact": {"run_id": Path(artifact_dir).name, "model_version": "N/A"},
            "data": {"latest_actual": "N/A", "source_age_days": "N/A"},
            "pipeline": {"logical_run_id": result["manifest"].get("logical_run_id", "N/A")},
            "forecast": {"target_date": "N/A", "forecast_id": "N/A"},
        }
        status_text = f"Status could not be generated: {status_error or 'unknown error'}"
    _send_email_or_log(
        subject=_success_subject(status),
        body=_success_body(result, status, status_text),
        env=env,
        transport_factory=transport_factory,
        logger=logger,
    )
    return result
