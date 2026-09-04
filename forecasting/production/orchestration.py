"""Scheduler-ready daily orchestration for the frozen XGBoost artifact."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from forecasting.paths import FORECASTS_DIR, MONITORING_DIR, OPERATIONS_DIR, TARGET_PANEL_5Y_PATH
from forecasting.production.data_refresh import (
    SOURCE_DATASET_ID, SOURCE_IDENTIFIER, SOURCE_PATH, refresh_production_data, seattle_today,
    target_panel_sha256, validate_target_panel_for_artifact,
)
from forecasting.production.inference import generate_forecast, load_verified_artifact
from forecasting.production.monitoring import discover_forecast_snapshots, run_monitoring
from forecasting.production.xgboost import MODEL_NAME, MODEL_VERSION, file_sha256


LOGGER = logging.getLogger(__name__)


PRODUCTION_RUN_LOCK_STALE_AGE_SECONDS = 3600


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ProductionRunLock:
    """Atomic lock file with conservative stale-lock recovery for ACA job retries.

    The Azure Container Apps Job replica timeout is 1200 seconds.  We default the
    stale threshold to 3600 seconds so a normally running production job cannot be
    classified as stale before the platform timeout has already terminated it.
    """
    def __init__(
        self, path: Path, *, stale_age_seconds: int = PRODUCTION_RUN_LOCK_STALE_AGE_SECONDS,
        owner_id: str | None = None, now_provider=_utcnow,
    ):
        self.path = path
        self.stale_age_seconds = stale_age_seconds
        self.owner_id = owner_id or uuid.uuid4().hex
        self.now_provider = now_provider
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._write_lockfile()
        except FileExistsError:
            self._recover_stale_lock_if_safe()
            try:
                self._write_lockfile()
            except FileExistsError as exc:
                details = self.path.read_text(encoding="utf-8") if self.path.exists() else "unavailable"
                raise RuntimeError(f"Another production run holds lock {self.path}: {details}") from exc
        self.acquired = True

    def release(self) -> None:
        if self.acquired and self.path.exists():
            self.path.unlink()
        self.acquired = False

    def _write_lockfile(self) -> None:
        with self.path.open("x", encoding="utf-8") as handle:
            json.dump(self._lock_payload(), handle)

    def _lock_payload(self) -> dict:
        return {
            "pid": os.getpid(),
            "acquired_at_utc": self.now_provider().isoformat(),
            "owner_id": self.owner_id,
            "hostname": socket.gethostname(),
        }

    def _recover_stale_lock_if_safe(self) -> None:
        existing = self._read_existing_lock()
        age_seconds = (self.now_provider() - existing["acquired_at"]).total_seconds()
        if age_seconds < 0 or age_seconds < self.stale_age_seconds:
            details = self.path.read_text(encoding="utf-8") if self.path.exists() else "unavailable"
            raise RuntimeError(f"Another production run holds lock {self.path}: {details}")
        LOGGER.warning(
            "Recovered stale production run lock %s owner_id=%s pid=%s acquired_at_utc=%s age_seconds=%.3f threshold_seconds=%s",
            self.path,
            existing["owner_id"],
            existing["pid"],
            existing["acquired_at_utc"],
            age_seconds,
            self.stale_age_seconds,
        )
        try:
            self.path.unlink()
        except FileNotFoundError:
            return

    def _read_existing_lock(self) -> dict:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(f"Another production run holds lock {self.path}: unavailable") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Another production run holds lock {self.path}: malformed or unreadable lock file; failing closed."
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Another production run holds lock {self.path}: malformed lock payload; failing closed.")
        pid = payload.get("pid")
        acquired_at_utc = payload.get("acquired_at_utc")
        owner_id = payload.get("owner_id")
        if not isinstance(pid, int) or not isinstance(acquired_at_utc, str) or not isinstance(owner_id, str) or not owner_id:
            raise RuntimeError(f"Another production run holds lock {self.path}: malformed lock payload; failing closed.")
        try:
            acquired_at = datetime.fromisoformat(acquired_at_utc)
        except ValueError as exc:
            raise RuntimeError(
                f"Another production run holds lock {self.path}: malformed acquired_at_utc; failing closed."
            ) from exc
        if acquired_at.tzinfo is None:
            raise RuntimeError(f"Another production run holds lock {self.path}: naive acquired_at_utc; failing closed.")
        return {"pid": pid, "owner_id": owner_id, "acquired_at_utc": acquired_at_utc, "acquired_at": acquired_at}


def _stable_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def _logical_run_id(artifact: dict, complete_date: pd.Timestamp, fingerprint: str) -> str:
    value = f"{artifact['metadata']['artifact_run_id']}|{complete_date.date()}|{fingerprint}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _previous_and_missed(artifact: dict, forecasts_root: str | Path, complete_date: pd.Timestamp) -> tuple[pd.Timestamp | None, list[str]]:
    snapshots = discover_forecast_snapshots(artifact=artifact, forecasts_root=forecasts_root)
    existing_dates = sorted({item.target_date.normalize() for item in snapshots})
    prior = [date for date in existing_dates if date <= complete_date]
    if not prior:
        return None, []
    last = prior[-1]
    # Those dates already have actuals at this run time and must never be backfilled.
    missed = pd.date_range(last + pd.Timedelta(days=1), complete_date, freq="D")
    return last, [date.date().isoformat() for date in missed]


def run_daily_pipeline(
    *, artifact_dir: str | Path, target_panel_path: str | Path = TARGET_PANEL_5Y_PATH,
    forecasts_root: str | Path = FORECASTS_DIR, monitoring_root: str | Path = MONITORING_DIR,
    operations_root: str | Path = OPERATIONS_DIR, max_source_age_days: int | None = None,
    skip_source_refresh: bool = False, refresh_function=refresh_production_data,
) -> dict:
    """Execute one logical state transition.  This function never trains or promotes."""
    started = time.monotonic()
    execution_id = uuid.uuid4().hex
    phase_statuses: list[dict] = []
    operations_base = Path(operations_root) / MODEL_NAME / MODEL_VERSION
    lock = ProductionRunLock(operations_base / ".daily_pipeline.lock", owner_id=execution_id)
    manifest: dict = {"execution_id": execution_id, "started_at_utc": _utcnow().isoformat(), "overall_status": "failed", "warnings": [], "errors": [], "phase_statuses": phase_statuses}

    def phase(name: str, action):
        start = time.monotonic()
        try:
            result = action()
            phase_statuses.append({"phase": name, "status": "completed", "elapsed_seconds": round(time.monotonic() - start, 6)})
            return result
        except Exception as exc:
            phase_statuses.append({"phase": name, "status": "failed", "elapsed_seconds": round(time.monotonic() - start, 6), "error": str(exc)})
            raise

    try:
        phase("acquire_run_lock", lock.acquire)
        artifact = phase("validate_artifact", lambda: load_verified_artifact(artifact_dir))
        manifest["artifact"] = {key: artifact["metadata"][key] for key in ["model_name", "model_version", "model_config_id", "artifact_run_id"]}
        manifest["artifact"]["pipeline_sha256"] = artifact["checksums"]["files"]["pipeline.joblib"]
        if skip_source_refresh:
            def local_refresh():
                panel = pd.read_parquet(target_panel_path)
                complete = panel["target_date"].max()
                panel = validate_target_panel_for_artifact(panel, artifact["baseline"]["expected_neighborhoods"], complete)
                return {"target_panel": panel, "summary": {"refresh_method": "skip_source_refresh_explicit_local_input", "source_identifier": SOURCE_IDENTIFIER, "source_dataset_id": SOURCE_DATASET_ID, "latest_source_date": pd.Timestamp(complete).date().isoformat(), "seattle_today": seattle_today().date().isoformat(), "latest_allowed_complete_date": (seattle_today()-pd.Timedelta(days=1)).date().isoformat(), "selected_complete_through_date": pd.Timestamp(complete).date().isoformat(), "target_panel_path": str(target_panel_path), "target_panel_sha256": target_panel_sha256(panel)}}
            refreshed = phase("load_explicit_local_target_panel", local_refresh)
            manifest["warnings"].append("Source refresh was explicitly skipped; local target panel was validated.")
        else:
            refreshed = phase("refresh_source_and_target_panel", lambda: refresh_function(expected_neighborhoods=artifact["baseline"]["expected_neighborhoods"], target_panel_path=target_panel_path))
        panel, refresh = refreshed["target_panel"], refreshed["summary"]
        complete = pd.Timestamp(refresh["selected_complete_through_date"]).normalize()
        source_age_days = int((seattle_today() - complete).days)
        refresh["source_data_age_days"] = source_age_days
        manifest["source"] = {"source_dataset_id": SOURCE_DATASET_ID, "source_identifier": SOURCE_IDENTIFIER, "source_path": SOURCE_PATH, "refresh_status": "skipped_explicitly" if skip_source_refresh else "successful", "latest_source_date": refresh["latest_source_date"], "complete_through_date": complete.date().isoformat(), "source_data_age_days": source_age_days}
        manifest["target_panel"] = {"path": str(target_panel_path), "sha256": refresh["target_panel_sha256"], "first_date": refresh.get("target_panel_start", panel.target_date.min().date().isoformat()), "last_date": refresh.get("target_panel_end", panel.target_date.max().date().isoformat()), "row_count": int(len(panel)), "neighborhood_count": int(panel.neighborhood.nunique())}
        if max_source_age_days is not None and source_age_days > max_source_age_days:
            raise ValueError(f"source_too_stale: source data age {source_age_days} exceeds max_source_age_days={max_source_age_days}.")
        if max_source_age_days is None and source_age_days > 0:
            manifest["warnings"].append(f"Source data are {source_age_days} days behind Seattle current date; no SLA threshold was supplied.")
        phase("validate_target_panel", lambda: validate_target_panel_for_artifact(panel, artifact["baseline"]["expected_neighborhoods"], complete))
        previous, missed_dates = phase("identify_missed_forecasts", lambda: _previous_and_missed(artifact, forecasts_root, complete))
        # Existing genuine forecasts are evaluated before a new prediction is made.
        existing_snapshots = discover_forecast_snapshots(artifact=artifact, forecasts_root=forecasts_root)
        pre_monitoring = phase(
            "evaluate_matured_forecasts",
            lambda: run_monitoring(artifact_dir=artifact_dir, target_panel=panel, forecasts_root=forecasts_root, monitoring_root=monitoring_root, update_latest=False)
            if existing_snapshots else None,
        )
        result = phase("generate_next_forecast", lambda: generate_forecast(artifact_dir=artifact_dir, target_panel=panel, forecast_origin=complete.isoformat(), output_root=forecasts_root))
        monitoring = phase("update_monitoring", lambda: run_monitoring(artifact_dir=artifact_dir, target_panel=panel, forecasts_root=forecasts_root, monitoring_root=monitoring_root))
        forecast = result["diagnostics"]
        manifest["forecast"] = {"action": "already_exists" if result["idempotent"] else "created", "forecast_id": forecast["forecast_id"], "forecast_origin": forecast["forecast_origin"], "target_date": forecast["target_date"], "neighborhood_count": int(len(result["forecast"]))}
        manifest["monitoring"] = {"forecasts_matured": int(monitoring["summary"]["n_matured"]), "forecasts_evaluated": int(monitoring["summary"]["n_evaluated"]), "monitoring_run_id": monitoring["report_dir"].name.removeprefix("run_id="), "path": str(monitoring["report_dir"]), "latest_realized_target_date": monitoring["summary"]["latest_realized_target_date"]}
        manifest["operations"] = {"missed_forecast_dates": missed_dates, "n_missed_forecast_dates": len(missed_dates), "previous_genuine_forecast_target_date": None if previous is None else previous.date().isoformat()}
        logical_run_id = _logical_run_id(artifact, complete, refresh["target_panel_sha256"])
        manifest["logical_run_id"] = logical_run_id
        manifest["completed_at_utc"] = _utcnow().isoformat()
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 6)
        manifest["overall_status"] = "completed"
        final_dir = operations_base / "runs" / f"logical_run_id={logical_run_id}"
        if final_dir.exists():
            existing = json.loads((final_dir / "run_manifest.json").read_text(encoding="utf-8"))
            return {"manifest": existing, "run_dir": final_dir, "idempotent": True, "forecast": result, "monitoring": monitoring}
        staging = final_dir.with_name(f".{final_dir.name}.{execution_id}.staging")
        staging.mkdir(parents=True, exist_ok=False)
        (staging / "run_manifest.json").write_text(_stable_json(manifest), encoding="utf-8")
        (staging / "refresh_summary.json").write_text(_stable_json(refresh), encoding="utf-8")
        checksums = {name: file_sha256(staging / name) for name in ["run_manifest.json", "refresh_summary.json"]}
        (staging / "checksums.json").write_text(_stable_json({"algorithm": "sha256", "files": checksums}), encoding="utf-8")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(final_dir)
        return {"manifest": manifest, "run_dir": final_dir, "idempotent": False, "forecast": result, "monitoring": monitoring}
    except Exception as exc:
        manifest["errors"].append(str(exc))
        manifest["completed_at_utc"] = _utcnow().isoformat()
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 6)
        # Failure records are deliberately separate from completed logical runs.
        failure_dir = operations_base / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        (failure_dir / f"execution_id={execution_id}.json").write_text(_stable_json(manifest), encoding="utf-8")
        raise
    finally:
        lock.release()
