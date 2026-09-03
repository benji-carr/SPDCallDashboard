"""Read-only status reporting for the locked production XGBoost artifact."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from forecasting.paths import FORECASTS_DIR, MONITORING_DIR, OPERATIONS_DIR
from forecasting.production.xgboost import MODEL_NAME, MODEL_VERSION


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _latest_path(paths: list[Path], timestamp_key: str | None = None) -> Path | None:
    if not paths:
        return None
    if timestamp_key:
        def sort_key(path: Path) -> tuple[pd.Timestamp, str]:
            try:
                value = _read_json(path).get(timestamp_key)
                return (pd.Timestamp(value) if value else pd.Timestamp.min, str(path))
            except (OSError, ValueError, json.JSONDecodeError):
                return (pd.Timestamp.min, str(path))
        return max(paths, key=sort_key)
    return max(paths, key=lambda path: (path.stat().st_mtime, str(path)))


def _read_latest_operation(operations_root: Path, artifact_run_id: str) -> dict[str, Any] | None:
    manifests: list[Path] = []
    for path in operations_root.glob(f"{MODEL_NAME}/{MODEL_VERSION}/runs/logical_run_id=*/run_manifest.json"):
        try:
            if _read_json(path).get("artifact", {}).get("artifact_run_id") == artifact_run_id:
                manifests.append(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    path = _latest_path(manifests, "completed_at_utc")
    if not path:
        return None
    manifest = _read_json(path)
    refresh_path = path.with_name("refresh_summary.json")
    manifest["refresh"] = _read_json(refresh_path) if refresh_path.is_file() else {}
    return manifest


def _read_latest_monitoring_report(monitoring_root: Path, artifact_run_id: str) -> tuple[dict[str, Any], Path] | tuple[None, None]:
    reports = list((monitoring_root / MODEL_NAME / MODEL_VERSION / f"artifact_run_id={artifact_run_id}" / "reports").glob("run_id=*/monitoring_summary.json"))
    valid = []
    for report in reports:
        try:
            if _read_json(report).get("artifact_run_id") == artifact_run_id:
                valid.append(report)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    path = _latest_path(valid, "generation_timestamp_utc")
    return (_read_json(path), path.parent) if path else (None, None)


def _latest_forecast(forecasts_root: Path, artifact_run_id: str) -> dict[str, Any] | None:
    root = forecasts_root / MODEL_NAME / MODEL_VERSION / "snapshots"
    candidates: list[tuple[pd.Timestamp, Path, pd.DataFrame]] = []
    for path in root.glob(f"target_date=*/artifact_run_id={artifact_run_id}/forecast.parquet"):
        frame = pd.read_parquet(path)
        if frame.empty or "artifact_run_id" not in frame or not (frame["artifact_run_id"] == artifact_run_id).all():
            continue
        candidates.append((pd.Timestamp(frame["target_date"].iloc[0]), path, frame))
    if not candidates:
        return None
    _, path, frame = max(candidates, key=lambda item: (item[0], str(item[1])))
    predictions = pd.to_numeric(frame["predicted_calls"], errors="raise")
    top10 = frame.nsmallest(10, "predicted_rank")
    return {
        "path": str(path.parent),
        "target_date": _date(frame["target_date"].iloc[0]),
        "forecast_origin": _date(frame["forecast_origin"].iloc[0]),
        "forecast_id": str(frame["forecast_id"].iloc[0]),
        "neighborhood_count": int(len(frame)),
        "prediction_mean": float(predictions.mean()),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "top10_predicted_calls": float(pd.to_numeric(top10["predicted_calls"]).sum()),
        "top10_share": float(pd.to_numeric(top10["predicted_calls"]).sum() / predictions.sum()) if predictions.sum() else None,
    }


def _compact_date_ranges(values: list[Any]) -> str:
    dates = sorted({pd.Timestamp(value).date() for value in values if value is not None})
    if not dates:
        return "NONE"
    ranges: list[tuple[date, date]] = []
    start = end = dates[0]
    for value in dates[1:]:
        if (value - end).days == 1:
            end = value
        else:
            ranges.append((start, end))
            start = end = value
    ranges.append((start, end))
    rendered = [a.isoformat() if a == b else f"{a.isoformat()} through {b.isoformat()}" for a, b in ranges]
    return "; ".join(rendered[:4]) + (f"; +{len(rendered) - 4} more ranges" if len(rendered) > 4 else "")


def _rows_by_window(frame: pd.DataFrame | None) -> dict[int, dict[str, Any]]:
    if frame is None or frame.empty:
        return {}
    result = {}
    for window in (7, 28, 90):
        subset = frame[frame["window_days_requested"] == window]
        if not subset.empty:
            result[window] = subset.sort_values("target_date").iloc[-1].to_dict()
    return result


def load_production_status(
    artifact_dir: str | Path,
    *,
    forecasts_root: str | Path = FORECASTS_DIR,
    monitoring_root: str | Path = MONITORING_DIR,
    operations_root: str | Path = OPERATIONS_DIR,
) -> dict[str, Any]:
    """Load existing production outputs without changing the filesystem or network state."""
    artifact_dir = Path(artifact_dir)
    metadata_path = artifact_dir / "metadata.json"
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"Artifact directory does not exist: {artifact_dir}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Required artifact metadata is missing: {metadata_path}")
    metadata = _read_json(metadata_path)
    artifact_run_id = metadata.get("artifact_run_id")
    if not isinstance(artifact_run_id, str) or not artifact_run_id or artifact_dir.name != artifact_run_id:
        raise ValueError("Artifact identity is malformed: directory name must match metadata artifact_run_id.")
    if metadata.get("model_name") != MODEL_NAME or metadata.get("model_version") != MODEL_VERSION:
        raise ValueError("Artifact metadata is not for the production SPD neighborhood XGBoost model.")

    forecasts_root, monitoring_root, operations_root = map(Path, (forecasts_root, monitoring_root, operations_root))
    operation = _read_latest_operation(operations_root, artifact_run_id)
    monitoring, report_dir = _read_latest_monitoring_report(monitoring_root, artifact_run_id)
    forecast = _latest_forecast(forecasts_root, artifact_run_id)

    daily = rolling = feature_drift = prediction_drift = None
    if report_dir:
        daily = pd.read_parquet(report_dir / "daily_performance.parquet") if (report_dir / "daily_performance.parquet").is_file() else None
        rolling = pd.read_parquet(report_dir / "rolling_performance.parquet") if (report_dir / "rolling_performance.parquet").is_file() else None
        feature_drift = pd.read_parquet(report_dir / "feature_drift.parquet") if (report_dir / "feature_drift.parquet").is_file() else None
        prediction_drift = pd.read_parquet(report_dir / "prediction_drift.parquet") if (report_dir / "prediction_drift.parquet").is_file() else None

    source = (operation or {}).get("source", {})
    latest_actual = (monitoring or {}).get("latest_observed_target_date") or source.get("latest_source_date")
    source_age = source.get("source_data_age_days", (monitoring or {}).get("source_data_age_days"))
    latest_daily = daily.sort_values("target_date").iloc[-1].to_dict() if daily is not None and not daily.empty else None
    if latest_daily:
        metrics_path = Path(latest_daily["evaluation_dir"]) / "daily_metrics.json"
        if metrics_path.is_file():
            latest_daily.update(_read_json(metrics_path))
    return {
        "artifact": {"run_id": artifact_run_id, "path": str(artifact_dir), "model_version": metadata.get("model_version"), "expected_neighborhoods": metadata.get("n_neighborhoods")},
        "data": {"latest_actual": _date(latest_actual), "source_age_days": source_age},
        "pipeline": {
            "status": (operation or {}).get("overall_status"), "logical_run_id": (operation or {}).get("logical_run_id"),
            "completed_at_utc": (operation or {}).get("completed_at_utc"), "refresh_seconds": (operation or {}).get("refresh", {}).get("refresh_elapsed_seconds", (operation or {}).get("refresh", {}).get("elapsed_seconds")),
            "api_request_count": (operation or {}).get("refresh", {}).get("api_request_count"),
            "missed_dates": (operation or {}).get("operations", {}).get("missed_forecast_dates", []),
        },
        "forecast": forecast,
        "performance": latest_daily,
        "rolling": _rows_by_window(rolling),
        "monitoring": {
            "forecasts_evaluated": (monitoring or {}).get("n_evaluated"), "forecasts_awaiting_actuals": (monitoring or {}).get("n_awaiting_actuals"),
            "feature_drift": _rows_by_window(feature_drift), "prediction_drift": _rows_by_window(prediction_drift),
        },
    }


def _value(value: Any, fallback: str = "N/A") -> str:
    return fallback if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


def _metric(value: Any, digits: int = 3, percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}" + ("%" if percent else "")


def format_production_status(status: dict[str, Any], *, verbose: bool = False) -> str:
    lines = ["SPD NEIGHBORHOOD FORECASTING - MODEL STATUS", "=" * 60, "", "MODEL"]
    artifact = status["artifact"]
    lines += [f"Artifact:                  {artifact['run_id']}", f"Model version:             {_value(artifact['model_version'])}", f"Artifact path:             {artifact['path']}", "", "DATA"]
    lines += [f"Latest actual:             {_value(status['data']['latest_actual'])}", f"Source age:                {_value(status['data']['source_age_days'])} days", f"Expected neighborhoods:    {_value(artifact['expected_neighborhoods'])}", "", "PIPELINE"]
    pipeline = status["pipeline"]
    lines += [f"Last pipeline run status:  {_value(pipeline['status']).upper()}", f"Logical run ID:            {_value(pipeline['logical_run_id'])}", f"Pipeline completed:        {_value(pipeline['completed_at_utc'])}", f"API refresh time:          {_metric(pipeline['refresh_seconds'], 2)} seconds", f"API request count:         {_value(pipeline['api_request_count'])}"]
    missed = pipeline["missed_dates"]
    lines += [f"Missed forecast days:      {len(missed)}", f"Missed range:              {_compact_date_ranges(missed)}", "", "LATEST FORECAST"]
    forecast = status["forecast"]
    if not forecast:
        lines.append("Latest forecast:           NONE")
    else:
        lines += [f"Latest forecast:           {forecast['target_date']}", f"Forecast origin:           {forecast['forecast_origin']}", f"Forecast ID:               {forecast['forecast_id']}", f"Neighborhood count:        {forecast['neighborhood_count']}", f"Mean predicted calls:      {_metric(forecast['prediction_mean'])}", f"Predicted range:           {_metric(forecast['prediction_min'])} to {_metric(forecast['prediction_max'])}", f"Top-10 volume/share:       {_metric(forecast['top10_predicted_calls'])} / {_metric((forecast['top10_share'] or 0) * 100, 2, True)}"]
    lines += ["", "REALIZED FORECAST PERFORMANCE"]
    performance = status["performance"]
    if not performance:
        lines += ["Latest evaluated forecast: none", "Realized performance:      AWAITING ACTUALS"]
    else:
        lines += [f"Latest evaluated forecast: {_date(performance.get('target_date'))}", f"MAE / sMAPE / MASE:        {_metric(performance.get('mae'))} / {_metric(performance.get('smape'), 2, True)} / {_metric(performance.get('mase'))}", f"RMSE / bias / abs. bias:   {_metric(performance.get('rmse'))} / {_metric(performance.get('bias'))} / {_metric(performance.get('mean_absolute_bias'))}", f"Top-10 accuracy/capture:   {_metric(performance.get('top10_accuracy_pct'), 2, True)} / {_metric(performance.get('top10_volume_capture_pct'), 2, True)}", f"Rank correlation:          {_metric(performance.get('rank_correlation'))}"]
    lines += ["", "ROLLING PERFORMANCE", "Window    Status       MAE      sMAPE     MASE"]
    for window in (7, 28, 90):
        row = status["rolling"].get(window)
        if not row or not bool(row.get("window_complete")):
            lines.append(f"{window:>3} day    INCOMPLETE  -        -         -")
        else:
            lines.append(f"{window:>3} day    COMPLETE    {_metric(row.get('mae')):<8} {_metric(row.get('smape'), 2, True):<9} {_metric(row.get('mase'))}")
    lines += ["", "MONITORING"]
    monitor = status["monitoring"]
    lines += [f"Forecasts evaluated:       {_value(monitor['forecasts_evaluated'])}", f"Forecasts awaiting actuals:{_value(monitor['forecasts_awaiting_actuals']):>8}", "Window    Feature drift   Prediction drift"]
    for window in (7, 28, 90):
        feature = monitor["feature_drift"].get(window)
        prediction = monitor["prediction_drift"].get(window)
        feature_text = "COMPLETE" if feature and bool(feature.get("window_complete")) else "INCOMPLETE"
        prediction_text = "COMPLETE" if prediction and bool(prediction.get("window_complete")) else "INCOMPLETE"
        lines.append(f"{window:>3} day    {feature_text:<14}{prediction_text}")
    if verbose and forecast:
        lines += ["", f"Latest forecast snapshot:  {forecast['path']}"]
    lines += ["", "=" * 60]
    return "\n".join(lines)
