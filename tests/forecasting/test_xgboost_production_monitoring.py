import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting.features.xgboost import build_xgboost_feature_panel, prepare_target_panel
from forecasting.production.inference import generate_forecast
from forecasting.production.monitoring import actuals_sha256, calculate_population_stability_index, discover_forecast_snapshots, run_monitoring
from forecasting.production.xgboost import file_sha256, train_production_model


def make_target_panel(n_days: int = 140, neighborhoods=("A", "B", "C")) -> pd.DataFrame:
    rows = []
    for day, date in enumerate(pd.date_range("2024-01-01", periods=n_days, freq="D")):
        for offset, neighborhood in enumerate(neighborhoods):
            rows.append(
                {
                    "target_date": date,
                    "neighborhood": neighborhood,
                    "calls": float(10 + offset * 2 + (day % 7) + (day // 21)),
                }
            )
    return prepare_target_panel(pd.DataFrame(rows))


def make_environment():
    root = Path("tests") / "_tmp" / f"monitoring_{uuid.uuid4().hex}"
    target_panel = make_target_panel()
    feature_panel = build_xgboost_feature_panel(target_panel)
    artifact_dir = train_production_model(
        target_panel=target_panel,
        feature_panel=feature_panel,
        target_panel_path=Path("synthetic_target.parquet"),
        feature_panel_path=Path("synthetic_features.parquet"),
        output_root=root / "artifacts",
    )["artifact_dir"]
    forecasts_root = root / "forecasts"
    for origin in ["2024-04-29", "2024-04-30", "2024-05-01"]:
        generate_forecast(
            artifact_dir=artifact_dir,
            target_panel=target_panel.loc[
                target_panel["target_date"] <= pd.Timestamp(origin)
            ].copy(),
            forecast_origin=origin,
            output_root=forecasts_root,
        )
    return root, target_panel, artifact_dir, forecasts_root


def test_monitoring_filters_to_matching_artifact_run_id_and_rejects_bad_checksums():
    root, target_panel, artifact_dir, forecasts_root = make_environment()
    try:
        artifact = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        wrong_root = (
            forecasts_root
            / "spd_neighborhood_xgboost"
            / "v1"
            / "snapshots"
            / "target_date=2024-05-02"
            / "artifact_run_id=other_run"
        )
        wrong_root.mkdir(parents=True, exist_ok=True)
        source = (
            forecasts_root
            / "spd_neighborhood_xgboost"
            / "v1"
            / "snapshots"
            / "target_date=2024-05-02"
            / f"artifact_run_id={artifact['artifact_run_id']}"
        )
        forecast = pd.read_parquet(source / "forecast.parquet")
        features = pd.read_parquet(source / "inference_features.parquet")
        diagnostics = json.loads((source / "inference_diagnostics.json").read_text(encoding="utf-8"))
        target_date = pd.Timestamp(forecast["target_date"].iloc[0]).date()
        forecast_origin = pd.Timestamp(forecast["forecast_origin"].iloc[0]).date()
        other_forecast_id = __import__("hashlib").sha256(
            f"spd_neighborhood_xgboost|v1|other_run|{forecast_origin}|{target_date}".encode("utf-8")
        ).hexdigest()[:20]
        forecast["artifact_run_id"] = "other_run"
        features["forecast_id"] = other_forecast_id
        forecast["forecast_id"] = other_forecast_id
        diagnostics["artifact_run_id"] = "other_run"
        diagnostics["forecast_id"] = other_forecast_id
        forecast.to_parquet(wrong_root / "forecast.parquet", index=False)
        features.to_parquet(wrong_root / "inference_features.parquet", index=False)
        (wrong_root / "inference_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2),
            encoding="utf-8",
        )
        (wrong_root / "checksums.json").write_text(
            json.dumps(
                {
                    "algorithm": "sha256",
                    "files": {
                        "forecast.parquet": file_sha256(wrong_root / "forecast.parquet"),
                        "inference_features.parquet": file_sha256(wrong_root / "inference_features.parquet"),
                        "inference_diagnostics.json": file_sha256(wrong_root / "inference_diagnostics.json"),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        loaded = discover_forecast_snapshots(
            artifact={"metadata": artifact, "baseline": json.loads((artifact_dir / "monitoring_baseline.json").read_text(encoding="utf-8"))},
            forecasts_root=forecasts_root,
        )
        assert len(loaded) == 3

        forecast_file = source / "forecast.parquet"
        forecast = pd.read_parquet(forecast_file)
        forecast.loc[0, "predicted_calls"] += 1
        forecast.to_parquet(forecast_file, index=False)
        with pytest.raises(ValueError, match="Checksum validation failed"):
            discover_forecast_snapshots(
                artifact={"metadata": artifact, "baseline": json.loads((artifact_dir / "monitoring_baseline.json").read_text(encoding="utf-8"))},
                forecasts_root=forecasts_root,
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_monitoring_rejects_wrong_forecast_identity():
    root, _, artifact_dir, forecasts_root = make_environment()
    try:
        result = run_monitoring(
            artifact_dir=artifact_dir,
            target_panel=make_target_panel(),
            forecasts_root=forecasts_root,
            monitoring_root=root / "monitoring",
            update_latest=False,
        )
        assert result["summary"]["n_forecasts"] == 3
        snapshot = result["forecast_inventory"]["snapshot_dir"].iloc[0]
        forecast_path = Path(snapshot) / "forecast.parquet"
        forecast = pd.read_parquet(forecast_path)
        forecast["model_name"] = "wrong_model"
        forecast.to_parquet(forecast_path, index=False)
        checksums = json.loads((Path(snapshot) / "checksums.json").read_text(encoding="utf-8"))
        checksums["files"]["forecast.parquet"] = file_sha256(forecast_path)
        (Path(snapshot) / "checksums.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")
        with pytest.raises(ValueError, match="model_name mismatch"):
            run_monitoring(
                artifact_dir=artifact_dir,
                target_panel=make_target_panel(),
                forecasts_root=forecasts_root,
                monitoring_root=root / "monitoring_2",
                update_latest=False,
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_psi_is_deterministic_and_uses_training_histogram():
    reference = {
        "interior_bin_edges": [0.0, 1.0],
        "expected_proportions": [0.25, 0.50, 0.25],
    }
    observed = pd.Series([-1.0, 0.5, 2.0, 0.6])
    first = calculate_population_stability_index(observed, reference)
    second = calculate_population_stability_index(observed.sample(frac=1, random_state=42), reference)
    assert first == pytest.approx(second)
    expected = (
        (0.25 - 0.25) * np.log(0.25 / 0.25)
        + (0.50 - 0.50) * np.log(0.50 / 0.50)
        + (0.25 - 0.25) * np.log(0.25 / 0.25)
    )
    assert first == pytest.approx(expected)


def test_feature_drift_excludes_calendar_features_and_reports_incomplete_windows():
    root, target_panel, artifact_dir, forecasts_root = make_environment()
    try:
        result = run_monitoring(
            artifact_dir=artifact_dir,
            target_panel=target_panel.iloc[:-3].copy(),
            forecasts_root=forecasts_root,
            monitoring_root=root / "monitoring",
            update_latest=False,
        )
        feature_drift = result["feature_drift"]
        daily_calendar = feature_drift.loc[
            (feature_drift["window_days_requested"] == 1)
            & (feature_drift["feature_group"] == "calendar")
        ]
        assert set(daily_calendar["feature_name"]) == {
            "is_weekend",
            "week_of_year_sin",
            "week_of_year_cos",
        }
        assert daily_calendar["statistical_drift_excluded"].all()
        rolling_28 = feature_drift.loc[
            (feature_drift["window_days_requested"] == 28)
            & (feature_drift["feature_name"] == "calls_lag_1")
        ]
        assert not rolling_28.empty
        assert rolling_28["window_complete"].eq(False).all()
        assert rolling_28["n_forecast_dates_available"].max() == 3
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_prediction_drift_reports_daily_and_rolling_z_reference():
    root, target_panel, artifact_dir, forecasts_root = make_environment()
    try:
        result = run_monitoring(
            artifact_dir=artifact_dir,
            target_panel=target_panel,
            forecasts_root=forecasts_root,
            monitoring_root=root / "monitoring",
            update_latest=False,
        )
        prediction = result["prediction_drift"]
        daily = prediction.loc[prediction["window_days_requested"] == 1]
        assert len(daily) == 3
        assert (daily["predicted_rank_min"] == 1).all()
        assert (daily["predicted_rank_max"] == 3).all()
        assert daily["mean_absolute_prediction_z"].notna().all()
        rolling = prediction.loc[prediction["window_days_requested"] == 7]
        assert not rolling.empty
        assert rolling["n_forecast_dates_available"].max() == 3
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_immature_forecasts_and_incomplete_actual_cross_sections_are_not_scored():
    root, target_panel, artifact_dir, forecasts_root = make_environment()
    try:
        incomplete_actuals = target_panel.loc[target_panel["target_date"] <= pd.Timestamp("2024-05-01")].copy()
        incomplete_actuals = incomplete_actuals.loc[
            ~(
                (incomplete_actuals["target_date"] == pd.Timestamp("2024-05-01"))
                & (incomplete_actuals["neighborhood"] == "C")
            )
        ].copy()
        result = run_monitoring(
            artifact_dir=artifact_dir,
            target_panel=incomplete_actuals,
            forecasts_root=forecasts_root,
            monitoring_root=root / "monitoring",
            update_latest=False,
        )
        inventory = result["forecast_inventory"].set_index("target_date")
        assert inventory.loc[pd.Timestamp("2024-04-30"), "maturity_status"] == "matured"
        assert inventory.loc[pd.Timestamp("2024-05-01"), "maturity_status"] == "not_evaluable_yet"
        assert inventory.loc[pd.Timestamp("2024-05-02"), "maturity_status"] == "awaiting_actuals"
        assert len(result["latest_evaluations"]) == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_realized_evaluations_are_immutable_idempotent_and_revision_aware():
    root, target_panel, artifact_dir, forecasts_root = make_environment()
    try:
        initial_actuals = target_panel.loc[target_panel["target_date"] <= pd.Timestamp("2024-05-02")].copy()
        first = run_monitoring(
            artifact_dir=artifact_dir,
            target_panel=initial_actuals,
            forecasts_root=forecasts_root,
            monitoring_root=root / "monitoring",
            update_latest=False,
        )
        second = run_monitoring(
            artifact_dir=artifact_dir,
            target_panel=initial_actuals,
            forecasts_root=forecasts_root,
            monitoring_root=root / "monitoring",
            update_latest=False,
        )
        assert len(first["all_evaluations"]) == 3
        assert second["evaluation_idempotent_count"] == 3

        revised = initial_actuals.copy()
        revised.loc[
            (revised["target_date"] == pd.Timestamp("2024-05-01"))
            & (revised["neighborhood"] == "A"),
            "calls",
        ] += 5
        revised_run = run_monitoring(
            artifact_dir=artifact_dir,
            target_panel=revised,
            forecasts_root=forecasts_root,
            monitoring_root=root / "monitoring",
            update_latest=False,
        )
        evaluation_rows = revised_run["all_evaluations"].loc[
            revised_run["all_evaluations"]["target_date"] == pd.Timestamp("2024-05-01")
        ]
        assert len(evaluation_rows) == 2
        current_fingerprint = actuals_sha256(
            revised.loc[revised["target_date"] == pd.Timestamp("2024-05-01")].copy()
        )
        latest = revised_run["latest_evaluations"].set_index("target_date")
        assert latest.loc[pd.Timestamp("2024-05-01"), "actuals_sha256"] == current_fingerprint
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_realized_metrics_match_expected_daily_values_and_reference_deltas():
    root, target_panel, artifact_dir, forecasts_root = make_environment()
    try:
        result = run_monitoring(
            artifact_dir=artifact_dir,
            target_panel=target_panel.loc[target_panel["target_date"] <= pd.Timestamp("2024-05-02")].copy(),
            forecasts_root=forecasts_root,
            monitoring_root=root / "monitoring",
            update_latest=False,
        )
        evaluations = result["latest_evaluations"]
        assert {"mae", "rmse", "bias", "mase", "top10_accuracy_pct", "rank_correlation"} <= set(evaluations.columns)
        assert (evaluations["mae"] >= 0).all()
        assert (evaluations["rmse"] >= 0).all()
        rolling = result["rolling_performance"]
        assert not rolling.empty
        row = rolling.loc[rolling["window_days_requested"] == 7].iloc[-1]
        assert row["rolling_mase_minus_reference"] == pytest.approx(
            row["mase"] - result["summary"]["frozen_reference_metrics"]["final_holdout"]["mean_mase"]
        )
        assert row["top10_accuracy_delta_pct_points"] == pytest.approx(
            row["top10_accuracy_pct"]
            - result["summary"]["frozen_reference_metrics"]["ranking"]["top10_accuracy_pct"]
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_monitoring_writes_report_files_checksums_and_latest_layer():
    root, target_panel, artifact_dir, forecasts_root = make_environment()
    try:
        result = run_monitoring(
            artifact_dir=artifact_dir,
            target_panel=target_panel.loc[target_panel["target_date"] <= pd.Timestamp("2024-05-02")].copy(),
            forecasts_root=forecasts_root,
            monitoring_root=root / "monitoring",
            update_latest=True,
        )
        report_dir = result["report_dir"]
        latest_dir = result["latest_dir"]
        assert (report_dir / "forecast_inventory.parquet").is_file()
        assert (report_dir / "checksums.json").is_file()
        checksums = json.loads((report_dir / "checksums.json").read_text(encoding="utf-8"))
        for name, expected in checksums["files"].items():
            assert file_sha256(report_dir / name) == expected
        assert latest_dir is not None
        assert (latest_dir / "monitoring_summary.json").is_file()
        assert (latest_dir / "feature_drift.parquet").is_file()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_max_data_age_days_is_recorded_and_enforced():
    root, target_panel, artifact_dir, forecasts_root = make_environment()
    try:
        truncated = target_panel.loc[target_panel["target_date"] <= pd.Timestamp("2024-05-02")].copy()
        result = run_monitoring(
            artifact_dir=artifact_dir,
            target_panel=truncated,
            forecasts_root=forecasts_root,
            monitoring_root=root / "monitoring",
            update_latest=False,
        )
        assert result["summary"]["source_data_age_days"] is not None
        with pytest.raises(ValueError, match="max_data_age_days"):
            run_monitoring(
                artifact_dir=artifact_dir,
                target_panel=truncated,
                forecasts_root=forecasts_root,
                monitoring_root=root / "monitoring2",
                max_data_age_days=1,
                update_latest=False,
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)
