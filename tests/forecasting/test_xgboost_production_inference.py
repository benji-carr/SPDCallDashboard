import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from forecasting.features.xgboost import build_xgboost_feature_panel, prepare_target_panel
from forecasting.production.inference import build_future_features, generate_forecast
from forecasting.production.xgboost import train_production_model


def make_panels(n_days=80):
    rows = []
    for day, date in enumerate(pd.date_range("2024-01-01", periods=n_days, freq="D")):
        for index, neighborhood in enumerate(("A", "B")):
            rows.append({"target_date": date, "neighborhood": neighborhood, "calls": float(index + day % 9)})
    target = prepare_target_panel(pd.DataFrame(rows))
    return target, build_xgboost_feature_panel(target)


def test_future_features_match_historical_backtest_features():
    target, historical_features = make_panels()
    target_date = historical_features["target_date"].iloc[-5]
    expected = sorted(target["neighborhood"].unique())
    future, origin, produced_date = build_future_features(target, expected, target_date - pd.Timedelta(days=1))
    actual = historical_features.loc[historical_features["target_date"] == target_date].sort_values("neighborhood")
    assert produced_date == target_date
    assert origin == target_date - pd.Timedelta(days=1)
    model_columns = [column for column in future if column != "target_date"]
    pd.testing.assert_frame_equal(
        future[model_columns].reset_index(drop=True),
        actual[model_columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_forecast_snapshot_is_idempotent_and_rejects_entity_change():
    target, features = make_panels()
    root = Path("tests") / "_tmp" / f"inference_{uuid.uuid4().hex}"
    try:
        artifact = train_production_model(target_panel=target, feature_panel=features, target_panel_path=Path("target"), feature_panel_path=Path("features"), output_root=root)["artifact_dir"]
        output = root / "forecasts"
        first = generate_forecast(artifact_dir=artifact, target_panel=target, output_root=output)
        second = generate_forecast(artifact_dir=artifact, target_panel=target, output_root=output)
        assert first["forecast"]["predicted_rank"].tolist() == [1, 2]
        assert second["idempotent"] is True
        changed = target.copy()
        changed.loc[changed.index[0], "neighborhood"] = "unexpected"
        with pytest.raises(ValueError, match="neighborhood set"):
            generate_forecast(artifact_dir=artifact, target_panel=changed, output_root=output)
    finally:
        shutil.rmtree(root, ignore_errors=True)
