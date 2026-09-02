import json
import shutil
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from forecasting.features.xgboost import build_xgboost_feature_panel, prepare_target_panel
from forecasting.production.xgboost import (
    FEATURE_SET_NAME, MODEL_CONFIG_ID, MODEL_NAME, MODEL_VERSION,
    PRODUCTION_XGB_PARAMS, dataframe_fingerprint, train_production_model,
    validate_training_data,
)


def make_panels(n_days=80):
    rows = []
    for day, date in enumerate(pd.date_range("2024-01-01", periods=n_days, freq="D")):
        for index, neighborhood in enumerate(("A", "B")):
            rows.append({"target_date": date, "neighborhood": neighborhood, "calls": float(10 + index + day % 7)})
    target = prepare_target_panel(pd.DataFrame(rows))
    return target, build_xgboost_feature_panel(target)


def artifact_root():
    return Path("tests") / "_tmp" / f"production_{uuid.uuid4().hex}"


def test_production_spec_is_locked_and_single_threaded():
    assert MODEL_CONFIG_ID == "be7924a5110a"
    assert MODEL_VERSION == "v1"
    assert FEATURE_SET_NAME == "lags_rolling_calendar"
    assert PRODUCTION_XGB_PARAMS["n_jobs"] == 1


def test_validation_uses_latest_eligible_row_and_rejects_duplicates():
    target, features = make_panels()
    numeric = [column for column in features.columns if column.startswith("calls_")] + ["is_weekend", "week_of_year_sin", "week_of_year_cos"]
    training = validate_training_data(target, features, numeric)
    assert training["target_date"].max() == features["target_date"].max()
    duplicated = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_training_data(target, duplicated, numeric)


def test_validation_rejects_missing_or_non_finite_feature():
    target, features = make_panels()
    numeric = ["calls_lag_1"]
    with pytest.raises(ValueError, match="missing"):
        validate_training_data(target, features.drop(columns="calls_lag_1"), numeric)
    invalid = features.copy()
    invalid.loc[invalid.index[0], "calls_lag_1"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_training_data(target, invalid, numeric)


def test_fingerprint_is_deterministic_across_row_order():
    _, features = make_panels()
    frame = features[["target_date", "neighborhood", "calls", "calls_lag_1"]]
    assert dataframe_fingerprint(frame) == dataframe_fingerprint(frame.sample(frac=1, random_state=42))


def test_training_writes_valid_immutable_artifacts():
    target, features = make_panels()
    root = artifact_root()
    try:
        result = train_production_model(
            target_panel=target, feature_panel=features,
            target_panel_path=Path("synthetic_target.parquet"), feature_panel_path=Path("synthetic_features.parquet"),
            output_root=root,
        )
        artifact = result["artifact_dir"]
        expected = {"pipeline.joblib", "booster.ubj", "metadata.json", "feature_schema.json", "training_summary.json", "monitoring_baseline.json", "checksums.json"}
        assert expected <= {path.name for path in artifact.iterdir()}
        metadata = json.loads((artifact / "metadata.json").read_text())
        schema = json.loads((artifact / "feature_schema.json").read_text())
        checksums = json.loads((artifact / "checksums.json").read_text())
        assert metadata["model_name"] == MODEL_NAME
        assert metadata["model_config_id"] == MODEL_CONFIG_ID
        assert metadata["n_training_rows"] == len(features)
        assert schema["numeric_features"][-3:] == ["is_weekend", "week_of_year_sin", "week_of_year_cos"]
        assert result["summary"]["round_trip_validation_passed"] is True
        assert len(checksums["files"]) == 6
        assert np.isfinite(joblib.load(artifact / "pipeline.joblib").predict(features[["neighborhood", *schema["numeric_features"]]].head(2))).all()
    finally:
        shutil.rmtree(root, ignore_errors=True)
