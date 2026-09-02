from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.evaluation.top10 import evaluate_day, rank_day
from forecasting.features.calendar import build_calendar_features
from forecasting.features.xgboost import (
    CALENDAR_XGB_FEATURES,
    TARGET_COLUMN,
    TARGET_HISTORY_FEATURES,
    prepare_target_panel,
)
from forecasting.paths import FORECASTS_DIR, MONITORING_DIR, TARGET_PANEL_5Y_PATH
from forecasting.production.inference import load_verified_artifact
from forecasting.production.xgboost import (
    MODEL_CONFIG_ID,
    MODEL_NAME,
    MODEL_VERSION,
    file_sha256,
    write_json,
)


ROLLING_WINDOWS_DAYS = [7, 28, 90]
FORECAST_FILES = [
    "forecast.parquet",
    "inference_features.parquet",
    "inference_diagnostics.json",
    "checksums.json",
]
EVALUATION_FILES = [
    "actuals_used.parquet",
    "realized_predictions.parquet",
    "daily_metrics.json",
]
REPORT_FILES = [
    "forecast_inventory.parquet",
    "daily_performance.parquet",
    "rolling_performance.parquet",
    "feature_drift.parquet",
    "prediction_drift.parquet",
    "monitoring_summary.json",
]


@dataclass(frozen=True)
class ForecastSnapshot:
    snapshot_dir: Path
    forecast: pd.DataFrame
    features: pd.DataFrame
    diagnostics: dict
    checksums: dict
    artifact_run_id: str
    forecast_id: str
    forecast_origin: pd.Timestamp
    target_date: pd.Timestamp


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_date_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], errors="raise").dt.normalize()


def _stable_json_dumps(data: dict) -> str:
    return json.dumps(data, indent=2, sort_keys=True, default=str) + "\n"


def _write_atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(_stable_json_dumps(payload), encoding="utf-8")
    temp.replace(path)


def _canonical_row_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    canonical = frame.loc[:, columns].copy()
    for column in canonical.columns:
        if "date" in column:
            canonical[column] = pd.to_datetime(
                canonical[column], errors="raise"
            ).dt.strftime("%Y-%m-%d")
    canonical = canonical.sort_values(columns).reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update("|".join(columns).encode("utf-8"))
    digest.update(
        pd.util.hash_pandas_object(canonical, index=False)
        .to_numpy(dtype="uint64")
        .tobytes()
    )
    return digest.hexdigest()


def _safe_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _validate_directory_checksums(directory: Path, expected_files: list[str]) -> dict:
    missing = [name for name in expected_files if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Directory is incomplete: missing {missing} in {directory}")
    checksums = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
    files = checksums.get("files", {})
    for name in expected_files:
        if name == "checksums.json":
            continue
        expected = files.get(name)
        if expected is None:
            raise ValueError(f"checksums.json is missing an entry for {name} in {directory}")
        actual = file_sha256(directory / name)
        if actual != expected:
            raise ValueError(f"Checksum validation failed for {directory / name}")
    return checksums


def _expected_calendar_features(target_date: pd.Timestamp) -> dict[str, float]:
    calendar = build_calendar_features(target_date, target_date).iloc[0]
    return {name: float(calendar[name]) for name in CALENDAR_XGB_FEATURES}


def _stat_summary(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce")
    observed = numeric.dropna()
    if observed.empty:
        return {
            "count": int(len(values)),
            "missing_count": int(numeric.isna().sum()),
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p10": None,
            "p50": None,
            "p90": None,
        }
    return {
        "count": int(len(values)),
        "missing_count": int(numeric.isna().sum()),
        "mean": float(observed.mean()),
        "std": float(observed.std(ddof=0)),
        "min": float(observed.min()),
        "max": float(observed.max()),
        "p10": float(observed.quantile(0.10)),
        "p50": float(observed.quantile(0.50)),
        "p90": float(observed.quantile(0.90)),
    }


def calculate_population_stability_index(
    observed: pd.Series,
    reference_histogram: dict,
) -> float | None:
    numeric = pd.to_numeric(observed, errors="coerce").dropna()
    if numeric.empty:
        return None
    edges = np.asarray(reference_histogram["interior_bin_edges"], dtype=float)
    bins = np.concatenate(([-np.inf], edges, [np.inf]))
    observed_counts = np.histogram(numeric.to_numpy(dtype=float), bins=bins)[0]
    observed_props = observed_counts / len(numeric)
    expected_props = np.asarray(reference_histogram["expected_proportions"], dtype=float)
    psi = 0.0
    for observed_prop, expected_prop in zip(observed_props, expected_props, strict=True):
        if observed_prop == 0:
            continue
        if expected_prop == 0:
            psi += float(observed_prop * np.log(observed_prop / 1e-12))
        else:
            psi += float((observed_prop - expected_prop) * np.log(observed_prop / expected_prop))
    return float(psi)


def _artifact_identity_valid(metadata: dict) -> None:
    identity = (
        metadata.get("model_name"),
        metadata.get("model_version"),
        metadata.get("model_config_id"),
    )
    if identity != (MODEL_NAME, MODEL_VERSION, MODEL_CONFIG_ID):
        raise ValueError("Monitoring only supports the locked production XGBoost artifact.")


def discover_forecast_snapshots(
    *,
    artifact: dict,
    forecasts_root: str | Path = FORECASTS_DIR,
    allow_mismatched_artifact_run_id: bool = False,
) -> list[ForecastSnapshot]:
    root = Path(forecasts_root) / MODEL_NAME / MODEL_VERSION / "snapshots"
    if not root.exists():
        return []
    snapshots: list[ForecastSnapshot] = []
    expected_neighborhoods = sorted(artifact["baseline"]["expected_neighborhoods"])
    for forecast_path in sorted(root.glob("target_date=*/artifact_run_id=*")):
        checksums = _validate_directory_checksums(forecast_path, FORECAST_FILES)
        forecast = pd.read_parquet(forecast_path / "forecast.parquet")
        features = pd.read_parquet(forecast_path / "inference_features.parquet")
        diagnostics = json.loads(
            (forecast_path / "inference_diagnostics.json").read_text(encoding="utf-8")
        )
        if forecast.empty or features.empty:
            raise ValueError(f"Forecast snapshot is empty: {forecast_path}")
        forecast["target_date"] = _normalize_date_column(forecast, "target_date")
        forecast["forecast_origin"] = _normalize_date_column(forecast, "forecast_origin")
        features["target_date"] = _normalize_date_column(features, "target_date")
        features["forecast_origin"] = _normalize_date_column(features, "forecast_origin")
        forecast_id = str(forecast["forecast_id"].iloc[0])
        artifact_run_id = str(forecast["artifact_run_id"].iloc[0])
        target_date = pd.Timestamp(forecast["target_date"].iloc[0]).normalize()
        forecast_origin = pd.Timestamp(forecast["forecast_origin"].iloc[0]).normalize()
        target_date_from_path = pd.Timestamp(
            forecast_path.parent.name.split("=", 1)[1]
        ).normalize()
        artifact_run_id_from_path = forecast_path.name.split("=", 1)[1]
        if target_date_from_path != target_date:
            raise ValueError(f"Forecast target_date path mismatch in {forecast_path}")
        if artifact_run_id_from_path != artifact_run_id:
            raise ValueError(f"Forecast artifact_run_id path mismatch in {forecast_path}")
        expected_forecast_id = hashlib.sha256(
            f"{MODEL_NAME}|{MODEL_VERSION}|{artifact_run_id}|{forecast_origin.date()}|{target_date.date()}".encode(
                "utf-8"
            )
        ).hexdigest()[:20]
        if forecast_id != expected_forecast_id:
            raise ValueError(f"Forecast ID mismatch in {forecast_path}")
        if str(diagnostics.get("forecast_id")) != forecast_id:
            raise ValueError(f"Inference diagnostics forecast_id mismatch in {forecast_path}")
        if str(diagnostics.get("artifact_run_id")) != artifact_run_id:
            raise ValueError(f"Inference diagnostics artifact_run_id mismatch in {forecast_path}")
        if pd.Timestamp(diagnostics["target_date"]).normalize() != target_date:
            raise ValueError(f"Inference diagnostics target_date mismatch in {forecast_path}")
        if pd.Timestamp(diagnostics["forecast_origin"]).normalize() != forecast_origin:
            raise ValueError(f"Inference diagnostics forecast_origin mismatch in {forecast_path}")
        if len(forecast) != len(expected_neighborhoods):
            raise ValueError(f"Unexpected forecast neighborhood count in {forecast_path}")
        actual_neighborhoods = sorted(forecast["neighborhood"].astype(str).tolist())
        if actual_neighborhoods != expected_neighborhoods:
            raise ValueError(f"Unexpected forecast neighborhood set in {forecast_path}")
        if forecast.duplicated(["forecast_id", "neighborhood"]).any():
            raise ValueError(f"Duplicate forecast rows in {forecast_path}")
        if features.duplicated(["forecast_id", "neighborhood"]).any():
            raise ValueError(f"Duplicate feature rows in {forecast_path}")
        if sorted(features["neighborhood"].astype(str).tolist()) != expected_neighborhoods:
            raise ValueError(f"Unexpected feature neighborhood set in {forecast_path}")
        if not allow_mismatched_artifact_run_id and artifact_run_id != artifact["metadata"]["artifact_run_id"]:
            continue
        for column in ["model_name", "model_version", "model_config_id"]:
            if forecast[column].nunique() != 1 or forecast[column].iloc[0] != artifact["metadata"][column]:
                raise ValueError(f"Forecast {column} mismatch in {forecast_path}")
        snapshots.append(
            ForecastSnapshot(
                snapshot_dir=forecast_path,
                forecast=forecast.sort_values("neighborhood").reset_index(drop=True),
                features=features.sort_values("neighborhood").reset_index(drop=True),
                diagnostics=diagnostics,
                checksums=checksums,
                artifact_run_id=artifact_run_id,
                forecast_id=forecast_id,
                forecast_origin=forecast_origin,
                target_date=target_date,
            )
        )
    return snapshots


def _per_neighborhood_baseline(baseline: dict) -> pd.DataFrame:
    return pd.DataFrame(baseline["per_neighborhood"]).sort_values("neighborhood").reset_index(drop=True)


def summarize_feature_drift(
    snapshots: list[ForecastSnapshot],
    baseline: dict,
) -> pd.DataFrame:
    rows: list[dict] = []
    numeric_baseline = baseline["numeric_features"]
    for snapshot in snapshots:
        features = snapshot.features.copy()
        for feature in CALENDAR_XGB_FEATURES:
            expected_value = _expected_calendar_features(snapshot.target_date)[feature]
            observed_values = pd.to_numeric(features[feature], errors="raise")
            if not np.allclose(observed_values.to_numpy(dtype=float), expected_value):
                raise ValueError(f"Calendar feature {feature} failed deterministic validation for {snapshot.snapshot_dir}")
            stats = _stat_summary(observed_values)
            rows.append(
                {
                    "forecast_id": snapshot.forecast_id,
                    "artifact_run_id": snapshot.artifact_run_id,
                    "target_date": snapshot.target_date,
                    "forecast_origin": snapshot.forecast_origin,
                    "window_days_requested": 1,
                    "n_forecast_dates_available": 1,
                    "window_complete": True,
                    "n_rows": len(features),
                    "feature_name": feature,
                    "feature_group": "calendar",
                    "statistical_drift_excluded": True,
                    "calendar_value": expected_value,
                    "training_mean": numeric_baseline[feature]["mean"],
                    "training_std": numeric_baseline[feature]["std"],
                    "standardized_mean_shift": None,
                    "below_training_min_count": int((observed_values < numeric_baseline[feature]["min"]).sum()),
                    "below_training_min_rate": float((observed_values < numeric_baseline[feature]["min"]).mean()),
                    "above_training_max_count": int((observed_values > numeric_baseline[feature]["max"]).sum()),
                    "above_training_max_rate": float((observed_values > numeric_baseline[feature]["max"]).mean()),
                    "psi": None,
                    **stats,
                }
            )
        for feature in TARGET_HISTORY_FEATURES:
            values = pd.to_numeric(features[feature], errors="coerce")
            reference = numeric_baseline[feature]
            stats = _stat_summary(values)
            training_std = float(reference["std"])
            mean_shift = None
            if stats["mean"] is not None and training_std > 0:
                mean_shift = float((stats["mean"] - reference["mean"]) / training_std)
            rows.append(
                {
                    "forecast_id": snapshot.forecast_id,
                    "artifact_run_id": snapshot.artifact_run_id,
                    "target_date": snapshot.target_date,
                    "forecast_origin": snapshot.forecast_origin,
                    "window_days_requested": 1,
                    "n_forecast_dates_available": 1,
                    "window_complete": True,
                    "n_rows": len(features),
                    "feature_name": feature,
                    "feature_group": "target_history",
                    "statistical_drift_excluded": False,
                    "calendar_value": None,
                    "training_mean": float(reference["mean"]),
                    "training_std": training_std,
                    "standardized_mean_shift": mean_shift,
                    "below_training_min_count": int((values < reference["min"]).sum()),
                    "below_training_min_rate": float((values < reference["min"]).mean()),
                    "above_training_max_count": int((values > reference["max"]).sum()),
                    "above_training_max_rate": float((values > reference["max"]).mean()),
                    "psi": calculate_population_stability_index(values, reference["histogram"])
                    if "histogram" in reference else None,
                    **stats,
                }
            )
    for window_days in ROLLING_WINDOWS_DAYS:
        for current in snapshots:
            eligible = [
                snapshot
                for snapshot in snapshots
                if current.target_date - pd.Timedelta(days=window_days - 1)
                <= snapshot.target_date
                <= current.target_date
            ]
            combined = pd.concat([snapshot.features for snapshot in eligible], ignore_index=True)
            available_dates = sorted({snapshot.target_date for snapshot in eligible})
            for feature in TARGET_HISTORY_FEATURES:
                values = pd.to_numeric(combined[feature], errors="coerce")
                reference = numeric_baseline[feature]
                stats = _stat_summary(values)
                training_std = float(reference["std"])
                mean_shift = None
                if stats["mean"] is not None and training_std > 0:
                    mean_shift = float((stats["mean"] - reference["mean"]) / training_std)
                rows.append(
                    {
                        "forecast_id": current.forecast_id,
                        "artifact_run_id": current.artifact_run_id,
                        "target_date": current.target_date,
                        "forecast_origin": current.forecast_origin,
                        "window_days_requested": window_days,
                        "n_forecast_dates_available": len(available_dates),
                        "window_complete": len(available_dates) == window_days,
                        "n_rows": len(combined),
                        "feature_name": feature,
                        "feature_group": "target_history",
                        "statistical_drift_excluded": False,
                        "calendar_value": None,
                        "training_mean": float(reference["mean"]),
                        "training_std": training_std,
                        "standardized_mean_shift": mean_shift,
                        "below_training_min_count": int((values < reference["min"]).sum()),
                        "below_training_min_rate": float((values < reference["min"]).mean()),
                        "above_training_max_count": int((values > reference["max"]).sum()),
                        "above_training_max_rate": float((values > reference["max"]).mean()),
                        "psi": calculate_population_stability_index(values, reference["histogram"])
                        if "histogram" in reference else None,
                        **stats,
                    }
                )
    if not rows:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["target_date", "window_days_requested", "feature_name"]
    ).reset_index(drop=True)


def summarize_prediction_drift(
    snapshots: list[ForecastSnapshot],
    baseline: dict,
) -> pd.DataFrame:
    neighborhood_baseline = _per_neighborhood_baseline(baseline)
    rows: list[dict] = []
    for snapshot in snapshots:
        forecast = snapshot.forecast.copy().sort_values("neighborhood").reset_index(drop=True)
        merged = forecast.merge(neighborhood_baseline, on="neighborhood", validate="one_to_one")
        z_score = (
            (merged["predicted_calls"] - merged["training_target_mean"])
            / merged["training_target_std"].replace({0.0: np.nan})
        )
        rows.append(
            {
                "forecast_id": snapshot.forecast_id,
                "artifact_run_id": snapshot.artifact_run_id,
                "target_date": snapshot.target_date,
                "forecast_origin": snapshot.forecast_origin,
                "window_days_requested": 1,
                "n_forecast_dates_available": 1,
                "window_complete": True,
                "n_rows": len(forecast),
                "prediction_mean": float(forecast["predicted_calls"].mean()),
                "prediction_std": float(forecast["predicted_calls"].std(ddof=0)),
                "prediction_min": float(forecast["predicted_calls"].min()),
                "prediction_max": float(forecast["predicted_calls"].max()),
                "prediction_p10": float(forecast["predicted_calls"].quantile(0.10)),
                "prediction_p50": float(forecast["predicted_calls"].quantile(0.50)),
                "prediction_p90": float(forecast["predicted_calls"].quantile(0.90)),
                "top10_predicted_calls": float(forecast.nsmallest(10, "predicted_rank")["predicted_calls"].sum()),
                "top10_share_of_total_predicted_volume": float(
                    forecast.nsmallest(10, "predicted_rank")["predicted_calls"].sum()
                    / forecast["predicted_calls"].sum()
                ),
                "predicted_rank_min": int(forecast["predicted_rank"].min()),
                "predicted_rank_max": int(forecast["predicted_rank"].max()),
                "mean_absolute_prediction_z": _safe_float(np.nanmean(np.abs(z_score))),
                "prediction_z_gt_2_count": int((np.abs(z_score) > 2).fillna(False).sum()),
                "prediction_z_gt_2_rate": float((np.abs(z_score) > 2).fillna(False).mean()),
                "prediction_z_gt_3_count": int((np.abs(z_score) > 3).fillna(False).sum()),
                "prediction_z_gt_3_rate": float((np.abs(z_score) > 3).fillna(False).mean()),
            }
        )
    for window_days in ROLLING_WINDOWS_DAYS:
        for current in snapshots:
            eligible = [
                snapshot
                for snapshot in snapshots
                if current.target_date - pd.Timedelta(days=window_days - 1)
                <= snapshot.target_date
                <= current.target_date
            ]
            combined = pd.concat(
                [snapshot.forecast[["neighborhood", "predicted_calls", "predicted_rank"]] for snapshot in eligible],
                ignore_index=True,
            )
            available_dates = sorted({snapshot.target_date for snapshot in eligible})
            merged = combined.merge(neighborhood_baseline, on="neighborhood", validate="many_to_one")
            z_score = (
                (merged["predicted_calls"] - merged["training_target_mean"])
                / merged["training_target_std"].replace({0.0: np.nan})
            )
            rows.append(
                {
                    "forecast_id": current.forecast_id,
                    "artifact_run_id": current.artifact_run_id,
                    "target_date": current.target_date,
                    "forecast_origin": current.forecast_origin,
                    "window_days_requested": window_days,
                    "n_forecast_dates_available": len(available_dates),
                    "window_complete": len(available_dates) == window_days,
                    "n_rows": len(combined),
                    "prediction_mean": float(combined["predicted_calls"].mean()),
                    "prediction_std": float(combined["predicted_calls"].std(ddof=0)),
                    "prediction_min": float(combined["predicted_calls"].min()),
                    "prediction_max": float(combined["predicted_calls"].max()),
                    "prediction_p10": float(combined["predicted_calls"].quantile(0.10)),
                    "prediction_p50": float(combined["predicted_calls"].quantile(0.50)),
                    "prediction_p90": float(combined["predicted_calls"].quantile(0.90)),
                    "top10_predicted_calls": None,
                    "top10_share_of_total_predicted_volume": None,
                    "predicted_rank_min": int(combined["predicted_rank"].min()),
                    "predicted_rank_max": int(combined["predicted_rank"].max()),
                    "mean_absolute_prediction_z": _safe_float(np.nanmean(np.abs(z_score))),
                    "prediction_z_gt_2_count": int((np.abs(z_score) > 2).fillna(False).sum()),
                    "prediction_z_gt_2_rate": float((np.abs(z_score) > 2).fillna(False).mean()),
                    "prediction_z_gt_3_count": int((np.abs(z_score) > 3).fillna(False).sum()),
                    "prediction_z_gt_3_rate": float((np.abs(z_score) > 3).fillna(False).mean()),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["target_date", "window_days_requested"]
    ).reset_index(drop=True)


def _prepare_actuals_panel(target_panel: pd.DataFrame) -> pd.DataFrame:
    actuals = prepare_target_panel(target_panel)
    actuals["calls"] = pd.to_numeric(actuals["calls"], errors="raise")
    if not np.isfinite(actuals["calls"].to_numpy(dtype=float)).all():
        raise ValueError("Actual target panel contains non-finite values.")
    return actuals


def actuals_sha256(actuals_for_date: pd.DataFrame) -> str:
    return _canonical_row_hash(
        actuals_for_date.rename(columns={"calls": "actual_calls"}),
        ["target_date", "neighborhood", "actual_calls"],
    )


def _validate_actual_cross_section(actuals_for_date: pd.DataFrame, expected_neighborhoods: list[str]) -> str | None:
    if actuals_for_date.duplicated(["target_date", "neighborhood"]).any():
        raise ValueError("Actual cross-section contains duplicate target_date/neighborhood rows.")
    neighborhoods = sorted(actuals_for_date["neighborhood"].astype(str).tolist())
    if len(actuals_for_date) != len(expected_neighborhoods) or neighborhoods != sorted(expected_neighborhoods):
        return "incomplete_actual_cross_section"
    if not np.isfinite(actuals_for_date["calls"].to_numpy(dtype=float)).all():
        raise ValueError("Actual cross-section contains non-finite calls values.")
    return None


def determine_forecast_maturity(
    snapshots: list[ForecastSnapshot],
    target_panel: pd.DataFrame,
    expected_neighborhoods: list[str],
) -> pd.DataFrame:
    actuals = _prepare_actuals_panel(target_panel)
    latest_actual_date = actuals["target_date"].max() if not actuals.empty else pd.NaT
    rows = []
    for snapshot in snapshots:
        status = "awaiting_actuals"
        reason = "target_date_after_latest_actual_date"
        current_actuals = actuals.loc[actuals["target_date"] == snapshot.target_date].copy()
        if pd.notna(latest_actual_date) and snapshot.target_date <= latest_actual_date:
            reason = _validate_actual_cross_section(current_actuals, expected_neighborhoods)
            if reason is None:
                status = "matured"
                reason = "complete_actual_cross_section"
            else:
                status = "not_evaluable_yet"
        rows.append(
            {
                "forecast_id": snapshot.forecast_id,
                "artifact_run_id": snapshot.artifact_run_id,
                "target_date": snapshot.target_date,
                "forecast_origin": snapshot.forecast_origin,
                "maturity_status": status,
                "maturity_reason": reason,
                "latest_observed_target_date": latest_actual_date,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["forecast_id", "target_date", "maturity_status", "maturity_reason", "latest_observed_target_date"])
    return pd.DataFrame(rows).sort_values("target_date").reset_index(drop=True)


def _build_realized_predictions(
    snapshot: ForecastSnapshot,
    actuals_for_date: pd.DataFrame,
    baseline: dict,
    metadata: dict,
) -> tuple[pd.DataFrame, dict]:
    actual_day = actuals_for_date.rename(columns={"calls": "actual"}).copy()
    actual_day["prediction"] = snapshot.forecast.set_index("neighborhood").loc[
        actual_day["neighborhood"], "predicted_calls"
    ].to_numpy()
    actual_day["fold"] = 0
    ranked = rank_day(actual_day)
    evaluated = evaluate_day(actual_day, top_k=10)
    denominators = _per_neighborhood_baseline(baseline)[
        ["neighborhood", "lag_7_seasonal_naive_mae_denominator"]
    ]
    realized = (
        snapshot.forecast.merge(
            ranked[["neighborhood", "actual", "actual_rank"]],
            on="neighborhood",
            validate="one_to_one",
        )
        .merge(denominators, on="neighborhood", validate="one_to_one")
        .sort_values("neighborhood")
        .reset_index(drop=True)
    )
    if realized["lag_7_seasonal_naive_mae_denominator"].isna().any() or (
        realized["lag_7_seasonal_naive_mae_denominator"] <= 0
    ).any():
        raise ValueError("Invalid zero or missing MASE denominator in monitoring baseline.")
    realized["error"] = realized["predicted_calls"] - realized["actual"]
    realized["absolute_error"] = realized["error"].abs()
    realized["squared_error"] = realized["error"] ** 2
    realized["mase_denominator"] = realized["lag_7_seasonal_naive_mae_denominator"]
    realized["scaled_absolute_error"] = realized["absolute_error"] / realized["mase_denominator"]
    realized["actual_calls"] = realized["actual"]
    realized["actuals_sha256"] = actuals_sha256(actuals_for_date)
    realized["model_name"] = metadata["model_name"]
    realized["model_version"] = metadata["model_version"]
    realized["model_config_id"] = metadata["model_config_id"]
    realized = realized[
        [
            "forecast_id",
            "artifact_run_id",
            "forecast_origin",
            "target_date",
            "neighborhood",
            "predicted_calls",
            "predicted_rank",
            "actual_calls",
            "actual_rank",
            "error",
            "absolute_error",
            "squared_error",
            "mase_denominator",
            "scaled_absolute_error",
            "model_name",
            "model_version",
            "model_config_id",
            "actuals_sha256",
        ]
    ]
    metrics = {
        "forecast_id": snapshot.forecast_id,
        "artifact_run_id": snapshot.artifact_run_id,
        "forecast_origin": snapshot.forecast_origin.date().isoformat(),
        "target_date": snapshot.target_date.date().isoformat(),
        "actuals_sha256": realized["actuals_sha256"].iloc[0],
        "n_neighborhoods": int(len(realized)),
        "mae": float(realized["absolute_error"].mean()),
        "rmse": float(np.sqrt(realized["squared_error"].mean())),
        "bias": float(realized["error"].mean()),
        "mean_absolute_bias": float(realized["error"].abs().mean()),
        "mase": float(realized["scaled_absolute_error"].mean()),
        "top10_accuracy_pct": float(100 * evaluated["top_k_accuracy"]),
        "mean_correct_top10": float(evaluated["overlap_count"]),
        "top10_volume_capture_pct": float(100 * evaluated["top10_volume_capture"]),
        "rank_correlation": float(evaluated["overall_rank_correlation"]),
    }
    return realized, metrics


def _write_immutable_directory(
    *,
    final_dir: Path,
    write_files,
    expected_files: list[str],
) -> bool:
    if final_dir.exists():
        checksums = _validate_directory_checksums(final_dir, [*expected_files, "checksums.json"])
        for name, writer in write_files.items():
            existing_hash = file_sha256(final_dir / name)
            temp = final_dir / f".compare.{name}"
            try:
                writer(temp)
                if file_sha256(temp) != existing_hash:
                    raise ValueError(f"Conflicting immutable content detected for {final_dir}")
            finally:
                if temp.exists():
                    temp.unlink()
        return True
    staging = final_dir.with_name(f".{final_dir.name}.staging")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    try:
        staging.mkdir(parents=True, exist_ok=False)
        for name, writer in write_files.items():
            writer(staging / name)
        _write_atomic_json(
            staging / "checksums.json",
            {
                "algorithm": "sha256",
                "files": {name: file_sha256(staging / name) for name in expected_files},
            },
        )
        staging.rename(final_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return False


def discover_existing_evaluations(monitoring_root: Path) -> pd.DataFrame:
    pattern = monitoring_root.glob("evaluations/target_date=*/forecast_id=*/actuals_sha256=*")
    rows = []
    for directory in sorted(pattern):
        _validate_directory_checksums(directory, [*EVALUATION_FILES, "checksums.json"])
        metrics = json.loads((directory / "daily_metrics.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "evaluation_dir": str(directory),
                "forecast_id": metrics["forecast_id"],
                "artifact_run_id": metrics["artifact_run_id"],
                "target_date": pd.Timestamp(metrics["target_date"]).normalize(),
                "forecast_origin": pd.Timestamp(metrics["forecast_origin"]).normalize(),
                "actuals_sha256": metrics["actuals_sha256"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "bias": metrics["bias"],
                "mean_absolute_bias": metrics["mean_absolute_bias"],
                "mase": metrics["mase"],
                "top10_accuracy_pct": metrics["top10_accuracy_pct"],
                "mean_correct_top10": metrics["mean_correct_top10"],
                "top10_volume_capture_pct": metrics["top10_volume_capture_pct"],
                "rank_correlation": metrics["rank_correlation"],
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "evaluation_dir",
                "forecast_id",
                "artifact_run_id",
                "target_date",
                "forecast_origin",
                "actuals_sha256",
                "mae",
                "rmse",
                "bias",
                "mean_absolute_bias",
                "mase",
                "top10_accuracy_pct",
                "mean_correct_top10",
                "top10_volume_capture_pct",
                "rank_correlation",
            ]
        )
    return pd.DataFrame(rows).sort_values(["target_date", "forecast_id", "actuals_sha256"]).reset_index(drop=True)


def select_latest_evaluations(evaluations: pd.DataFrame) -> pd.DataFrame:
    if evaluations.empty:
        return evaluations.copy()
    return (
        evaluations.sort_values(["forecast_id", "actuals_sha256"])
        .groupby("forecast_id", as_index=False)
        .tail(1)
        .sort_values("target_date")
        .reset_index(drop=True)
    )


def select_current_evaluations(
    evaluations: pd.DataFrame,
    current_actuals_fingerprints: dict[str, str],
) -> pd.DataFrame:
    if evaluations.empty:
        return evaluations.copy()
    mask = evaluations.apply(
        lambda row: current_actuals_fingerprints.get(str(row["forecast_id"]))
        == str(row["actuals_sha256"]),
        axis=1,
    )
    return evaluations.loc[mask].sort_values("target_date").reset_index(drop=True)


def build_rolling_performance(
    latest_evaluations: pd.DataFrame,
    reference_metrics: dict,
) -> pd.DataFrame:
    rows: list[dict] = []
    if latest_evaluations.empty:
        return pd.DataFrame()
    latest_evaluations = latest_evaluations.sort_values("target_date").reset_index(drop=True)
    for _, current in latest_evaluations.iterrows():
        target_date = pd.Timestamp(current["target_date"]).normalize()
        for window_days in ROLLING_WINDOWS_DAYS:
            mask = latest_evaluations["target_date"].between(
                target_date - pd.Timedelta(days=window_days - 1),
                target_date,
            )
            window = latest_evaluations.loc[mask].copy()
            if window.empty:
                continue
            rows.append(
                {
                    "forecast_id": current["forecast_id"],
                    "artifact_run_id": current["artifact_run_id"],
                    "target_date": target_date,
                    "window_days_requested": window_days,
                    "n_realized_dates_available": int(window["target_date"].nunique()),
                    "window_complete": int(window["target_date"].nunique()) == window_days,
                    "window_start": pd.Timestamp(window["target_date"].min()).date().isoformat(),
                    "window_end": pd.Timestamp(window["target_date"].max()).date().isoformat(),
                    "mae": float(window["mae"].mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(window["rmse"])))),
                    "bias": float(window["bias"].mean()),
                    "mean_absolute_bias": float(window["mean_absolute_bias"].mean()),
                    "mase": float(window["mase"].mean()),
                    "top10_accuracy_pct": float(window["top10_accuracy_pct"].mean()),
                    "mean_correct_top10": float(window["mean_correct_top10"].mean()),
                    "top10_volume_capture_pct": float(window["top10_volume_capture_pct"].mean()),
                    "rank_correlation": float(window["rank_correlation"].mean()),
                    "rolling_mase_minus_reference": float(
                        window["mase"].mean() - reference_metrics["final_holdout"]["mean_mase"]
                    ),
                    "rolling_mase_ratio_to_reference": float(
                        window["mase"].mean() / reference_metrics["final_holdout"]["mean_mase"]
                    ),
                    "rolling_mae_minus_reference": float(
                        window["mae"].mean() - reference_metrics["final_holdout"]["mean_mae"]
                    ),
                    "top10_accuracy_delta_pct_points": float(
                        window["top10_accuracy_pct"].mean()
                        - reference_metrics["ranking"]["top10_accuracy_pct"]
                    ),
                    "rank_correlation_delta": float(
                        window["rank_correlation"].mean()
                        - reference_metrics["ranking"]["rank_correlation"]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["target_date", "window_days_requested"]).reset_index(drop=True)


def run_monitoring(
    *,
    artifact_dir: str | Path,
    target_panel: pd.DataFrame | None = None,
    target_panel_path: str | Path = TARGET_PANEL_5Y_PATH,
    forecasts_root: str | Path = FORECASTS_DIR,
    monitoring_root: str | Path = MONITORING_DIR,
    allow_mismatched_artifact_run_id: bool = False,
    max_data_age_days: int | None = None,
    update_latest: bool = True,
) -> dict:
    artifact = load_verified_artifact(artifact_dir)
    _artifact_identity_valid(artifact["metadata"])
    snapshots = discover_forecast_snapshots(
        artifact=artifact,
        forecasts_root=forecasts_root,
        allow_mismatched_artifact_run_id=allow_mismatched_artifact_run_id,
    )
    monitoring_root = (
        Path(monitoring_root)
        / MODEL_NAME
        / MODEL_VERSION
        / f"artifact_run_id={artifact['metadata']['artifact_run_id']}"
    )
    expected_neighborhoods = artifact["baseline"]["expected_neighborhoods"]
    target_panel_df = (
        pd.read_parquet(target_panel_path) if target_panel is None else target_panel.copy()
    )
    actuals = _prepare_actuals_panel(target_panel_df)
    latest_observed_target_date = actuals["target_date"].max() if not actuals.empty else pd.NaT
    source_data_age_days = None
    if pd.notna(latest_observed_target_date):
        source_data_age_days = int(
            (pd.Timestamp(datetime.now(timezone.utc).date()) - latest_observed_target_date).days
        )
        if max_data_age_days is not None and source_data_age_days > max_data_age_days:
            raise ValueError(
                f"Source data age {source_data_age_days} exceeds max_data_age_days={max_data_age_days}."
            )
    feature_drift = summarize_feature_drift(snapshots, artifact["baseline"])
    prediction_drift = summarize_prediction_drift(snapshots, artifact["baseline"])
    maturity = determine_forecast_maturity(snapshots, actuals, expected_neighborhoods)
    inventory_rows = []
    for snapshot in snapshots:
        maturity_row = maturity.loc[maturity["forecast_id"] == snapshot.forecast_id].iloc[0]
        inventory_rows.append(
            {
                "forecast_id": snapshot.forecast_id,
                "artifact_run_id": snapshot.artifact_run_id,
                "snapshot_dir": str(snapshot.snapshot_dir),
                "forecast_origin": snapshot.forecast_origin,
                "target_date": snapshot.target_date,
                "maturity_status": maturity_row["maturity_status"],
                "maturity_reason": maturity_row["maturity_reason"],
                "latest_observed_target_date": maturity_row["latest_observed_target_date"],
                "source_data_age_days": source_data_age_days,
            }
        )
    forecast_inventory = pd.DataFrame(inventory_rows)
    if not forecast_inventory.empty:
        forecast_inventory = forecast_inventory.sort_values("target_date").reset_index(drop=True)

    evaluation_dirs: list[Path] = []
    evaluation_idempotent_count = 0
    for snapshot in snapshots:
        maturity_row = maturity.loc[maturity["forecast_id"] == snapshot.forecast_id].iloc[0]
        if maturity_row["maturity_status"] != "matured":
            continue
        actuals_for_date = actuals.loc[actuals["target_date"] == snapshot.target_date].copy()
        realized, metrics = _build_realized_predictions(
            snapshot, actuals_for_date, artifact["baseline"], artifact["metadata"]
        )
        evaluation_dir = (
            monitoring_root
            / "evaluations"
            / f"target_date={snapshot.target_date.date()}"
            / f"forecast_id={snapshot.forecast_id}"
            / f"actuals_sha256={metrics['actuals_sha256']}"
        )
        is_idempotent = _write_immutable_directory(
            final_dir=evaluation_dir,
            expected_files=EVALUATION_FILES,
            write_files={
                "actuals_used.parquet": lambda path, frame=actuals_for_date.assign(
                    actual_calls=actuals_for_date["calls"],
                    actuals_sha256=metrics["actuals_sha256"],
                )[["target_date", "neighborhood", "actual_calls", "actuals_sha256"]]: frame.to_parquet(path, index=False),
                "realized_predictions.parquet": lambda path, frame=realized: frame.to_parquet(path, index=False),
                "daily_metrics.json": lambda path, payload=metrics: path.write_text(
                    _stable_json_dumps(payload), encoding="utf-8"
                ),
            },
        )
        if is_idempotent:
            evaluation_idempotent_count += 1
        evaluation_dirs.append(evaluation_dir)

    all_evaluations = discover_existing_evaluations(monitoring_root)
    current_actuals_fingerprints = {}
    for snapshot in snapshots:
        maturity_row = maturity.loc[maturity["forecast_id"] == snapshot.forecast_id].iloc[0]
        if maturity_row["maturity_status"] == "matured":
            current_actuals_fingerprints[snapshot.forecast_id] = actuals_sha256(
                actuals.loc[actuals["target_date"] == snapshot.target_date].copy()
            )
    latest_evaluations = select_current_evaluations(
        all_evaluations, current_actuals_fingerprints
    )
    rolling_performance = build_rolling_performance(
        latest_evaluations, artifact["metadata"]["reference_evaluation_metrics"]
    )
    latest_realized_date = (
        latest_evaluations["target_date"].max() if not latest_evaluations.empty else pd.NaT
    )
    monitoring_run_id = hashlib.sha256(
        json.dumps(
            {
                "artifact_run_id": artifact["metadata"]["artifact_run_id"],
                "forecast_ids": sorted(forecast_inventory["forecast_id"].tolist()),
                "latest_actual_date": None
                if pd.isna(latest_observed_target_date)
                else latest_observed_target_date.date().isoformat(),
                "latest_evaluation_fingerprints": latest_evaluations[
                    ["forecast_id", "actuals_sha256"]
                ].astype(str).values.tolist()
                if not latest_evaluations.empty
                else [],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:20]
    report_dir = monitoring_root / "reports" / f"run_id={monitoring_run_id}"
    generation_timestamp = _utc_now_iso()
    if (report_dir / "monitoring_summary.json").is_file():
        existing_summary = json.loads(
            (report_dir / "monitoring_summary.json").read_text(encoding="utf-8")
        )
        generation_timestamp = existing_summary.get(
            "generation_timestamp_utc", generation_timestamp
        )
    summary = {
        "model_name": artifact["metadata"]["model_name"],
        "model_version": artifact["metadata"]["model_version"],
        "model_config_id": artifact["metadata"]["model_config_id"],
        "artifact_run_id": artifact["metadata"]["artifact_run_id"],
        "training_data_sha256": artifact["metadata"]["training_data_sha256"],
        "generation_timestamp_utc": generation_timestamp,
        "latest_forecast_target_date": None
        if forecast_inventory.empty
        else pd.Timestamp(forecast_inventory["target_date"].max()).date().isoformat(),
        "latest_observed_target_date": None
        if pd.isna(latest_observed_target_date)
        else latest_observed_target_date.date().isoformat(),
        "latest_realized_target_date": None
        if pd.isna(latest_realized_date)
        else pd.Timestamp(latest_realized_date).date().isoformat(),
        "source_data_age_days": source_data_age_days,
        "n_forecasts": int(len(forecast_inventory)),
        "n_matured": int((forecast_inventory["maturity_status"] == "matured").sum()) if not forecast_inventory.empty else 0,
        "n_awaiting_actuals": int((forecast_inventory["maturity_status"] == "awaiting_actuals").sum()) if not forecast_inventory.empty else 0,
        "n_incomplete_actuals": int((forecast_inventory["maturity_status"] == "not_evaluable_yet").sum()) if not forecast_inventory.empty else 0,
        "n_evaluated": int(len(latest_evaluations)),
        "target_history_features_monitored": TARGET_HISTORY_FEATURES,
        "calendar_features_excluded_from_daily_statistical_drift": CALENDAR_XGB_FEATURES,
        "frozen_reference_metrics": artifact["metadata"]["reference_evaluation_metrics"],
        "actual_fingerprint_method": "SHA-256 of sorted target_date/neighborhood/actual_calls rows using pandas.hash_pandas_object with normalized dates and fixed column order.",
        "automatic_retraining_or_promotion": False,
    }
    _write_immutable_directory(
        final_dir=report_dir,
        expected_files=REPORT_FILES,
        write_files={
            "forecast_inventory.parquet": lambda path, frame=forecast_inventory: frame.to_parquet(path, index=False),
            "daily_performance.parquet": lambda path, frame=latest_evaluations: frame.to_parquet(path, index=False),
            "rolling_performance.parquet": lambda path, frame=rolling_performance: frame.to_parquet(path, index=False),
            "feature_drift.parquet": lambda path, frame=feature_drift: frame.to_parquet(path, index=False),
            "prediction_drift.parquet": lambda path, frame=prediction_drift: frame.to_parquet(path, index=False),
            "monitoring_summary.json": lambda path, payload=summary: path.write_text(
                _stable_json_dumps(payload), encoding="utf-8"
            ),
        },
    )
    _validate_directory_checksums(report_dir, [*REPORT_FILES, "checksums.json"])
    latest_dir = monitoring_root / "latest"
    if update_latest:
        latest_stage = latest_dir.with_name(".latest.staging")
        if latest_stage.exists():
            shutil.rmtree(latest_stage, ignore_errors=True)
        latest_stage.mkdir(parents=True, exist_ok=False)
        try:
            shutil.copy2(report_dir / "monitoring_summary.json", latest_stage / "monitoring_summary.json")
            shutil.copy2(report_dir / "rolling_performance.parquet", latest_stage / "rolling_performance.parquet")
            shutil.copy2(report_dir / "feature_drift.parquet", latest_stage / "feature_drift.parquet")
            shutil.copy2(report_dir / "prediction_drift.parquet", latest_stage / "prediction_drift.parquet")
            if latest_dir.exists():
                shutil.rmtree(latest_dir)
            latest_stage.rename(latest_dir)
        except Exception:
            shutil.rmtree(latest_stage, ignore_errors=True)
            raise
    return {
        "artifact": artifact,
        "forecast_inventory": forecast_inventory,
        "feature_drift": feature_drift,
        "prediction_drift": prediction_drift,
        "maturity": maturity,
        "all_evaluations": all_evaluations,
        "latest_evaluations": latest_evaluations,
        "rolling_performance": rolling_performance,
        "report_dir": report_dir,
        "latest_dir": latest_dir if update_latest else None,
        "monitoring_root": monitoring_root,
        "evaluation_dirs": evaluation_dirs,
        "evaluation_idempotent_count": evaluation_idempotent_count,
        "summary": summary,
    }
