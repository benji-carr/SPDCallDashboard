import shutil
import uuid
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from forecasting.features.xgboost import build_xgboost_feature_panel, prepare_target_panel
from forecasting.production import inference
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


def test_forecast_source_age_uses_seattle_calendar_date_and_is_idempotent(monkeypatch):
    target, features = make_panels(n_days=80)
    target["target_date"] += pd.Timedelta(days=(pd.Timestamp("2026-08-31") - target["target_date"].max()).days)
    features["target_date"] += pd.Timedelta(days=(pd.Timestamp("2026-08-31") - features["target_date"].max()).days)
    root = Path("tests") / "_tmp" / f"inference_age_{uuid.uuid4().hex}"
    try:
        artifact = train_production_model(
            target_panel=target,
            feature_panel=features,
            target_panel_path=Path("target"),
            feature_panel_path=Path("features"),
            output_root=root,
        )["artifact_dir"]
        args = {
            "artifact_dir": artifact,
            "target_panel": target,
            "forecast_origin": "2026-08-31",
            "output_root": root / "forecasts",
            "as_of_date": "2026-09-03",
        }
        with pytest.raises(ValueError, match="Source data age 3 exceeds max_data_age_days=2"):
            generate_forecast(**args, max_data_age_days=2)
        first = generate_forecast(**args, max_data_age_days=3)
        second = generate_forecast(**args, max_data_age_days=3)
        monkeypatch.setattr(inference, "seattle_today", lambda: pd.Timestamp("2026-09-03"))
        default_as_of_args = {key: value for key, value in args.items() if key != "as_of_date"}
        default_as_of_args["output_root"] = root / "default_as_of_forecasts"
        default_as_of = generate_forecast(**default_as_of_args, max_data_age_days=3)
        persisted_diagnostics = __import__("json").loads(
            (first["snapshot"] / "inference_diagnostics.json").read_text(encoding="utf-8")
        )

        expected_id = hashlib.sha256(
            f"spd_neighborhood_xgboost|v1|{first['diagnostics']['artifact_run_id']}|2026-08-31|2026-09-01".encode("utf-8")
        ).hexdigest()[:20]
        assert first["diagnostics"]["source_data_age_days"] == 3
        assert persisted_diagnostics["source_data_age_days"] == 3
        assert first["diagnostics"]["target_date"] == "2026-09-01"
        assert first["diagnostics"]["forecast_id"] == expected_id
        assert second["idempotent"] is True
        assert second["diagnostics"]["forecast_id"] == expected_id
        assert default_as_of["diagnostics"]["source_data_age_days"] == 3
    finally:
        shutil.rmtree(root, ignore_errors=True)
