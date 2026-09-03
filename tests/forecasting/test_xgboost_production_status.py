import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from forecasting.production.status import format_production_status, load_production_status


ARTIFACT_ID = "20260902T002733Z"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_outputs(tmp_path: Path, *, with_performance: bool = True, complete_windows: bool = False) -> tuple[Path, dict]:
    artifact = tmp_path / "artifacts" / ARTIFACT_ID
    _write_json(artifact / "metadata.json", {"artifact_run_id": ARTIFACT_ID, "model_name": "spd_neighborhood_xgboost", "model_version": "v1", "n_neighborhoods": 58})
    operation = tmp_path / "operations" / "spd_neighborhood_xgboost" / "v1" / "runs" / "logical_run_id=logical-1"
    _write_json(operation / "run_manifest.json", {"artifact": {"artifact_run_id": ARTIFACT_ID}, "logical_run_id": "logical-1", "overall_status": "completed", "completed_at_utc": "2026-09-03T19:15:02+00:00", "source": {"latest_source_date": "2026-08-31", "source_data_age_days": 3}, "operations": {"missed_forecast_dates": ["2026-08-14", "2026-08-15", "2026-08-17"]}})
    _write_json(operation / "refresh_summary.json", {"refresh_elapsed_seconds": 254.64116, "api_request_count": 58})
    forecast_dir = tmp_path / "forecasts" / "spd_neighborhood_xgboost" / "v1" / "snapshots" / "target_date=2026-09-01" / f"artifact_run_id={ARTIFACT_ID}"
    forecast_dir.mkdir(parents=True)
    pd.DataFrame({"neighborhood": ["A", "B"], "predicted_calls": [10.0, 5.0], "predicted_rank": [1, 2], "forecast_id": ["forecast-1", "forecast-1"], "forecast_origin": ["2026-08-31", "2026-08-31"], "target_date": ["2026-09-01", "2026-09-01"], "artifact_run_id": [ARTIFACT_ID, ARTIFACT_ID]}).to_parquet(forecast_dir / "forecast.parquet")
    report = tmp_path / "monitoring" / "spd_neighborhood_xgboost" / "v1" / f"artifact_run_id={ARTIFACT_ID}" / "reports" / "run_id=report-1"
    _write_json(report / "monitoring_summary.json", {"artifact_run_id": ARTIFACT_ID, "generation_timestamp_utc": "2026-09-03T19:15:02+00:00", "latest_observed_target_date": "2026-08-31", "source_data_age_days": 3, "n_evaluated": 1 if with_performance else 0, "n_awaiting_actuals": 1})
    rows = [{"target_date": "2026-08-13", "window_days_requested": window, "window_complete": complete_windows, "mae": 3.5, "smape": 37.0, "mase": 0.77} for window in (7, 28, 90)]
    pd.DataFrame(rows).to_parquet(report / "rolling_performance.parquet")
    drift = pd.DataFrame([{"target_date": "2026-09-01", "window_days_requested": window, "window_complete": complete_windows} for window in (7, 28, 90)])
    drift.to_parquet(report / "feature_drift.parquet")
    drift.to_parquet(report / "prediction_drift.parquet")
    if with_performance:
        evaluation = report / "evaluation"
        _write_json(evaluation / "daily_metrics.json", {"smape": 37.1})
        pd.DataFrame([{"evaluation_dir": str(evaluation), "target_date": "2026-08-13", "mae": 3.526, "rmse": 5.449, "bias": -0.044, "mean_absolute_bias": 3.526, "mase": 0.777, "top10_accuracy_pct": 80.0, "top10_volume_capture_pct": 94.177, "rank_correlation": 0.935}]).to_parquet(report / "daily_performance.parquet")
    return artifact, {"forecasts": tmp_path / "forecasts", "monitoring": tmp_path / "monitoring", "operations": tmp_path / "operations"}


def _load(artifact: Path, roots: dict) -> str:
    return format_production_status(load_production_status(artifact, forecasts_root=roots["forecasts"], monitoring_root=roots["monitoring"], operations_root=roots["operations"]))


def test_status_report_displays_production_sections_and_metrics(tmp_path):
    artifact, roots = _make_outputs(tmp_path, complete_windows=True)
    report = _load(artifact, roots)
    for text in ("Artifact:                  20260902T002733Z", "Latest actual:             2026-08-31", "Source age:                3 days", "Last pipeline run status:  COMPLETED", "API refresh time:          254.64 seconds", "Missed forecast days:      3", "2026-08-14 through 2026-08-15; 2026-08-17", "Latest forecast:           2026-09-01", "Forecast ID:               forecast-1", "3.526 / 37.10% / 0.777", "  7 day    COMPLETE", "Forecasts awaiting actuals:       1"):
        assert text in report


def test_status_report_handles_absent_metrics_and_incomplete_windows(tmp_path):
    artifact, roots = _make_outputs(tmp_path, with_performance=False, complete_windows=False)
    report = _load(artifact, roots)
    assert "Latest evaluated forecast: none" in report
    assert "Realized performance:      AWAITING ACTUALS" in report
    assert report.count("INCOMPLETE") >= 9


def test_status_is_scoped_to_requested_artifact(tmp_path):
    artifact, roots = _make_outputs(tmp_path)
    other = roots["monitoring"] / "spd_neighborhood_xgboost" / "v1" / "artifact_run_id=other" / "reports" / "run_id=other"
    _write_json(other / "monitoring_summary.json", {"artifact_run_id": "other", "generation_timestamp_utc": "2099-01-01T00:00:00+00:00", "n_evaluated": 99})
    assert "Forecasts evaluated:       1" in _load(artifact, roots)


def test_status_is_read_only_and_missing_optional_telemetry_is_safe(tmp_path):
    artifact, roots = _make_outputs(tmp_path)
    (roots["operations"] / "spd_neighborhood_xgboost" / "v1" / "runs" / "logical_run_id=logical-1" / "refresh_summary.json").unlink()
    before = hashlib.sha256(b"".join(sorted(path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()))).hexdigest()
    report = _load(artifact, roots)
    after = hashlib.sha256(b"".join(sorted(path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()))).hexdigest()
    assert before == after
    assert "API refresh time:          - seconds" in report


def test_invalid_artifact_directory_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Artifact directory does not exist"):
        load_production_status(tmp_path / "missing")


def test_status_displays_failed_pipeline_and_no_forecast(tmp_path):
    artifact, roots = _make_outputs(tmp_path)
    manifest = roots["operations"] / "spd_neighborhood_xgboost" / "v1" / "runs" / "logical_run_id=logical-1" / "run_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["overall_status"] = "failed"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    for path in roots["forecasts"].rglob("forecast.parquet"):
        path.unlink()
    report = _load(artifact, roots)
    assert "Last pipeline run status:  FAILED" in report
    assert "Latest forecast:           NONE" in report


def test_malformed_artifact_identity_fails_clearly(tmp_path):
    artifact, _ = _make_outputs(tmp_path)
    metadata = artifact / "metadata.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["artifact_run_id"] = "other"
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Artifact identity is malformed"):
        load_production_status(artifact)
