import json
import logging
from pathlib import Path

import pytest

from forecasting.production.notifying_runner import run_daily_pipeline_with_notifications
from scripts.forecasting import run_xgboost_daily_pipeline_with_notifications as notify_cli


def _env() -> dict[str, str]:
    return {
        "SPD_XGBOOST_NOTIFICATION_ENABLED": "1",
        "SPD_XGBOOST_SMTP_HOST": "smtp.example.com",
        "SPD_XGBOOST_SMTP_PORT": "587",
        "SPD_XGBOOST_SMTP_USERNAME": "ops@example.com",
        "SPD_XGBOOST_SMTP_PASSWORD": "super-secret-password",
        "SPD_XGBOOST_EMAIL_FROM": "ops@example.com",
        "SPD_XGBOOST_EMAIL_TO": "team@example.com",
    }


def _status(*_args, **_kwargs):
    return {
        "artifact": {"run_id": "20260902T002733Z", "model_version": "v1"},
        "data": {"latest_actual": "2026-09-03", "source_age_days": 1},
        "pipeline": {"logical_run_id": "logical-123"},
        "forecast": {"target_date": "2026-09-04", "forecast_id": "forecast-123"},
    }


def _status_text(status, *, verbose: bool = False):
    assert verbose is False
    return (
        "SPD NEIGHBORHOOD FORECASTING - MODEL STATUS\n"
        "Artifact: 20260902T002733Z\n"
        "Forecast ID: forecast-123\n"
    )


class RecordingTransport:
    sent_messages = []

    def __init__(self, _config):
        self.__class__.sent_messages = []

    def send_email(self, *, subject: str, body: str) -> None:
        self.__class__.sent_messages.append({"subject": subject, "body": body})


class RaisingTransport:
    def __init__(self, _config, *, message: str = "smtp unavailable"):
        self.message = message

    def send_email(self, *, subject: str, body: str) -> None:
        raise RuntimeError(self.message)


def test_successful_pipeline_sends_success_email(tmp_path):
    def pipeline_runner(**_kwargs):
        return {
            "manifest": {"execution_id": "execution-123", "logical_run_id": "logical-123"},
            "run_dir": tmp_path / "operations" / "run",
            "idempotent": False,
        }

    result = run_daily_pipeline_with_notifications(
        artifact_dir=tmp_path / "artifacts" / "20260902T002733Z",
        forecasts_root=tmp_path / "forecasts",
        monitoring_root=tmp_path / "monitoring",
        operations_root=tmp_path / "operations",
        env=_env(),
        pipeline_runner=pipeline_runner,
        status_loader=_status,
        status_formatter=_status_text,
        transport_factory=RecordingTransport,
    )

    assert result["idempotent"] is False
    assert len(RecordingTransport.sent_messages) == 1
    email = RecordingTransport.sent_messages[0]
    assert "SUCCESS 20260902T002733Z 2026-09-04" in email["subject"]
    for text in (
        "Overall success: yes",
        "Execution ID: execution-123",
        "Logical run ID: logical-123",
        "Artifact ID: 20260902T002733Z",
        "Artifact version: v1",
        "Latest actual date: 2026-09-03",
        "Source age: 1 days",
        "Latest forecast target date: 2026-09-04",
        "Forecast ID: forecast-123",
        "Production run idempotent: no",
        "SPD NEIGHBORHOOD FORECASTING - MODEL STATUS",
    ):
        assert text in email["body"]


def test_successful_pipeline_preserves_success_when_notification_fails(tmp_path, caplog):
    def pipeline_runner(**_kwargs):
        return {
            "manifest": {"execution_id": "execution-123", "logical_run_id": "logical-123"},
            "run_dir": tmp_path / "operations" / "run",
            "idempotent": True,
        }

    with caplog.at_level(logging.ERROR):
        result = run_daily_pipeline_with_notifications(
            artifact_dir=tmp_path / "artifacts" / "20260902T002733Z",
            forecasts_root=tmp_path / "forecasts",
            monitoring_root=tmp_path / "monitoring",
            operations_root=tmp_path / "operations",
            env=_env(),
            pipeline_runner=pipeline_runner,
            status_loader=_status,
            status_formatter=_status_text,
            transport_factory=lambda config: RaisingTransport(config, message="smtp unavailable"),
        )

    assert result["idempotent"] is True
    assert "notification delivery failed" in caplog.text
    assert "smtp unavailable" in caplog.text


def test_failed_pipeline_sends_failure_email_and_reraises(tmp_path):
    operations_root = tmp_path / "operations"
    failures_dir = operations_root / "spd_neighborhood_xgboost" / "v1" / "failures"

    def pipeline_runner(**_kwargs):
        failures_dir.mkdir(parents=True, exist_ok=True)
        (failures_dir / "execution_id=execution-999.json").write_text(
            json.dumps(
                {
                    "execution_id": "execution-999",
                    "phase_statuses": [
                        {"phase": "refresh_source_and_target_panel", "status": "failed"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_daily_pipeline_with_notifications(
            artifact_dir=tmp_path / "artifacts" / "20260902T002733Z",
            forecasts_root=tmp_path / "forecasts",
            monitoring_root=tmp_path / "monitoring",
            operations_root=operations_root,
            env=_env(),
            pipeline_runner=pipeline_runner,
            status_loader=_status,
            status_formatter=_status_text,
            transport_factory=RecordingTransport,
        )

    assert len(RecordingTransport.sent_messages) == 1
    email = RecordingTransport.sent_messages[0]
    assert "FAILURE 20260902T002733Z execution-999" in email["subject"]
    for text in (
        "Overall success: no",
        "Execution ID: execution-999",
        "Failed phase: refresh_source_and_target_panel",
        "Exception class: RuntimeError",
        "Exception message: boom",
        "SPD NEIGHBORHOOD FORECASTING - MODEL STATUS",
    ):
        assert text in email["body"]


def test_failed_pipeline_includes_status_generation_fallback_when_needed(tmp_path):
    operations_root = tmp_path / "operations"
    failures_dir = operations_root / "spd_neighborhood_xgboost" / "v1" / "failures"

    def pipeline_runner(**_kwargs):
        failures_dir.mkdir(parents=True, exist_ok=True)
        (failures_dir / "execution_id=execution-404.json").write_text(
            json.dumps(
                {
                    "execution_id": "execution-404",
                    "phase_statuses": [{"phase": "update_monitoring", "status": "failed"}],
                }
            ),
            encoding="utf-8",
        )
        raise ValueError("monitoring broke")

    def broken_status_loader(*_args, **_kwargs):
        raise FileNotFoundError("status inputs missing")

    with pytest.raises(ValueError, match="monitoring broke"):
        run_daily_pipeline_with_notifications(
            artifact_dir=tmp_path / "artifacts" / "20260902T002733Z",
            forecasts_root=tmp_path / "forecasts",
            monitoring_root=tmp_path / "monitoring",
            operations_root=operations_root,
            env=_env(),
            pipeline_runner=pipeline_runner,
            status_loader=broken_status_loader,
            status_formatter=_status_text,
            transport_factory=RecordingTransport,
        )

    email = RecordingTransport.sent_messages[0]
    assert "Status could not be generated: FileNotFoundError: status inputs missing" in email["body"]


def test_notification_logs_redact_secrets(tmp_path, caplog):
    def pipeline_runner(**_kwargs):
        return {
            "manifest": {"execution_id": "execution-123", "logical_run_id": "logical-123"},
            "run_dir": tmp_path / "operations" / "run",
            "idempotent": False,
        }

    env = _env()
    secret = env["SPD_XGBOOST_SMTP_PASSWORD"]
    with caplog.at_level(logging.ERROR):
        run_daily_pipeline_with_notifications(
            artifact_dir=tmp_path / "artifacts" / "20260902T002733Z",
            forecasts_root=tmp_path / "forecasts",
            monitoring_root=tmp_path / "monitoring",
            operations_root=tmp_path / "operations",
            env=env,
            pipeline_runner=pipeline_runner,
            status_loader=_status,
            status_formatter=_status_text,
            transport_factory=lambda config: RaisingTransport(config, message=f"auth failed for {secret}"),
        )

    assert secret not in caplog.text
    assert "***REDACTED***" in caplog.text


def test_cli_returns_non_zero_when_pipeline_fails(monkeypatch):
    monkeypatch.setattr(
        notify_cli,
        "run_daily_pipeline_with_notifications",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("pipeline failed")),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_xgboost_daily_pipeline_with_notifications.py",
            "--artifact-dir",
            "artifact-dir",
        ],
    )
    assert notify_cli.main() == 1
