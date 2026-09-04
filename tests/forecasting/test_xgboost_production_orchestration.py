import json
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from forecasting.features.xgboost import build_xgboost_feature_panel, prepare_target_panel
from forecasting.production.data_refresh import (
    build_target_panel, complete_through_date, target_panel_sha256,
    validate_source_schema, validate_target_panel_for_artifact,
)
from forecasting.production.inference import generate_forecast
from forecasting.production.orchestration import (
    PRODUCTION_RUN_LOCK_STALE_AGE_SECONDS, ProductionRunLock, run_daily_pipeline,
)
from forecasting.production.xgboost import train_production_model


def make_panel(days=80, neighborhoods=("A", "B", "C")):
    rows = []
    for day, target_date in enumerate(pd.date_range("2024-01-01", periods=days, freq="D")):
        for offset, neighborhood in enumerate(neighborhoods):
            rows.append({"target_date": target_date, "neighborhood": neighborhood, "calls": float(day % 7 + day // 3 + offset)})
    return prepare_target_panel(pd.DataFrame(rows))


def make_environment():
    root = Path("tests") / "_tmp" / f"orchestration_{uuid.uuid4().hex}"
    panel = make_panel()
    artifact = train_production_model(target_panel=panel, feature_panel=build_xgboost_feature_panel(panel), target_panel_path=Path("target"), feature_panel_path=Path("features"), output_root=root / "artifacts")["artifact_dir"]
    return root, panel, artifact


def source_for(panel):
    rows = []
    for row in panel.itertuples(index=False):
        for number in range(int(row.calls)):
            rows.append({"cad_event_number": f"{row.target_date.date()}-{row.neighborhood}-{number}", "cad_event_original_time_queued": f"{row.target_date.date()}T12:00:00Z", "dispatch_neighborhood": row.neighborhood})
    return pd.DataFrame(rows)


def test_source_schema_and_distinct_cad_target_semantics():
    source = pd.DataFrame([
        {"cad_event_number": "1", "cad_event_original_time_queued": "2024-01-01T12:00:00Z", "dispatch_neighborhood": "A"},
        {"cad_event_number": "1", "cad_event_original_time_queued": "2024-01-01T12:00:00Z", "dispatch_neighborhood": "A"},
        {"cad_event_number": "2", "cad_event_original_time_queued": "2024-01-01T12:00:00Z", "dispatch_neighborhood": "A"},
    ])
    panel = build_target_panel(source, expected_neighborhoods=["A", "B"], selected_complete_through_date="2024-01-01")
    assert panel.loc[(panel.neighborhood.eq("A")) & (panel.target_date.eq("2024-01-01")), "calls"].iloc[0] == 2
    assert panel.loc[(panel.neighborhood.eq("B")) & (panel.target_date.eq("2024-01-01")), "calls"].iloc[0] == 0
    with pytest.raises(ValueError, match="required columns"):
        validate_source_schema(source.drop(columns="cad_event_number"))


def test_complete_day_excludes_seattle_current_day():
    source = pd.DataFrame([
        {"cad_event_number": "old", "cad_event_original_time_queued": "2024-03-09T20:00:00Z", "dispatch_neighborhood": "A"},
        {"cad_event_number": "current", "cad_event_original_time_queued": "2024-03-10T20:00:00Z", "dispatch_neighborhood": "A"},
    ])
    result = complete_through_date(source, now=datetime(2024, 3, 10, 15, tzinfo=ZoneInfo("America/Los_Angeles")))
    assert result["selected_complete_through_date"] == pd.Timestamp("2024-03-09")


def test_panel_validation_fingerprint_and_entity_failures():
    panel = make_panel(days=40, neighborhoods=("A", "B"))
    assert target_panel_sha256(panel) == target_panel_sha256(panel.sample(frac=1, random_state=7))
    validate_target_panel_for_artifact(panel, ["A", "B"], panel.target_date.max())
    with pytest.raises(ValueError, match="entity set"):
        validate_target_panel_for_artifact(panel.loc[panel.neighborhood.eq("A")], ["A", "B"], panel.target_date.max())
    negative = panel.copy(); negative.loc[0, "calls"] = -1
    with pytest.raises(ValueError, match="negative"):
        validate_target_panel_for_artifact(negative, ["A", "B"], panel.target_date.max())


def test_refresh_failure_does_not_fall_back_to_local_panel():
    root, panel, artifact = make_environment()
    try:
        local = root / "panel.parquet"; panel.to_parquet(local, index=False)
        def fail(**kwargs): raise RuntimeError("network unavailable")
        with pytest.raises(RuntimeError, match="network unavailable"):
            run_daily_pipeline(artifact_dir=artifact, target_panel_path=local, operations_root=root / "operations", refresh_function=fail)
        assert list((root / "forecasts").rglob("forecast.parquet")) == []
        assert list((root / "operations" / "spd_neighborhood_xgboost" / "v1" / "failures").glob("*.json"))
    finally: shutil.rmtree(root, ignore_errors=True)


def test_pipeline_generates_one_next_unknown_forecast_and_is_idempotent():
    root, panel, artifact = make_environment()
    try:
        local = root / "panel.parquet"; panel.to_parquet(local, index=False)
        args = dict(artifact_dir=artifact, target_panel_path=local, forecasts_root=root / "forecasts", monitoring_root=root / "monitoring", operations_root=root / "operations", skip_source_refresh=True)
        first = run_daily_pipeline(**args)
        assert first["manifest"]["forecast"]["target_date"] == "2024-03-21"
        second = run_daily_pipeline(**args)
        assert second["idempotent"] is True
        assert len(list((root / "forecasts").rglob("forecast.parquet"))) == 1
    finally: shutil.rmtree(root, ignore_errors=True)


def test_matured_forecast_evaluated_and_missed_dates_not_backfilled():
    root, panel, artifact = make_environment()
    try:
        forecasts = root / "forecasts"
        origin = pd.Timestamp("2024-03-10")
        generate_forecast(artifact_dir=artifact, target_panel=panel.loc[panel.target_date <= origin], forecast_origin=origin, output_root=forecasts)
        extended = make_panel(days=80, neighborhoods=("A", "B", "C")); local = root / "panel.parquet"; extended.to_parquet(local, index=False)
        result = run_daily_pipeline(artifact_dir=artifact, target_panel_path=local, forecasts_root=forecasts, monitoring_root=root / "monitoring", operations_root=root / "operations", skip_source_refresh=True)
        assert result["manifest"]["monitoring"]["forecasts_evaluated"] == 1
        assert result["manifest"]["operations"]["n_missed_forecast_dates"] > 0
        snapshots = list(forecasts.rglob("forecast.parquet"))
        assert len(snapshots) == 2
        assert result["manifest"]["forecast"]["target_date"] == "2024-03-21"
    finally: shutil.rmtree(root, ignore_errors=True)


def test_freshness_gate_and_lock_release_after_failure():
    root, panel, artifact = make_environment()
    try:
        local = root / "panel.parquet"; panel.to_parquet(local, index=False)
        with pytest.raises(ValueError, match="source_too_stale"):
            run_daily_pipeline(artifact_dir=artifact, target_panel_path=local, operations_root=root / "operations", skip_source_refresh=True, max_source_age_days=0)
        production_lock = root / "operations" / "spd_neighborhood_xgboost" / "v1" / ".daily_pipeline.lock"
        assert not production_lock.exists()
        lock = ProductionRunLock(root / "lock")
        lock.acquire()
        with pytest.raises(RuntimeError, match="holds lock"):
            ProductionRunLock(root / "lock").acquire()
        lock.release()
        second = ProductionRunLock(root / "lock")
        second.acquire()
        second.release()
    finally: shutil.rmtree(root, ignore_errors=True)


def test_production_run_lock_records_metadata_and_releases():
    root = Path("tests") / "_tmp" / f"orchestration_lock_{uuid.uuid4().hex}"
    try:
        lock = ProductionRunLock(root / "lock", owner_id="execution-123")
        lock.acquire()
        payload = json.loads((root / "lock").read_text(encoding="utf-8"))
        assert payload["pid"] > 0
        assert payload["owner_id"] == "execution-123"
        assert payload["hostname"]
        assert datetime.fromisoformat(payload["acquired_at_utc"]).tzinfo is not None
        lock.release()
        assert not (root / "lock").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_second_active_lock_acquisition_fails():
    root = Path("tests") / "_tmp" / f"orchestration_lock_{uuid.uuid4().hex}"
    try:
        first = ProductionRunLock(root / "lock", owner_id="first")
        first.acquire()
        with pytest.raises(RuntimeError, match="holds lock"):
            ProductionRunLock(root / "lock", owner_id="second").acquire()
    finally:
        first.release()
        shutil.rmtree(root, ignore_errors=True)


def test_stale_lock_is_recovered(caplog):
    root = Path("tests") / "_tmp" / f"orchestration_lock_{uuid.uuid4().hex}"
    try:
        lock_path = root / "lock"
        acquired_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        recovered_at = acquired_at + timedelta(seconds=PRODUCTION_RUN_LOCK_STALE_AGE_SECONDS + 5)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({"pid": 111, "acquired_at_utc": acquired_at.isoformat(), "owner_id": "abandoned", "hostname": "worker-a"}), encoding="utf-8")
        lock = ProductionRunLock(lock_path, owner_id="retry", now_provider=lambda: recovered_at)
        with caplog.at_level("WARNING"):
            lock.acquire()
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["owner_id"] == "retry"
        assert "Recovered stale production run lock" in caplog.text
    finally:
        lock.release()
        shutil.rmtree(root, ignore_errors=True)


def test_malformed_lock_fails_closed():
    root = Path("tests") / "_tmp" / f"orchestration_lock_{uuid.uuid4().hex}"
    try:
        lock_path = root / "lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("{not-json", encoding="utf-8")
        with pytest.raises(RuntimeError, match="failing closed"):
            ProductionRunLock(lock_path).acquire()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stale_lock_recovery_still_fails_if_another_process_wins(monkeypatch):
    root = Path("tests") / "_tmp" / f"orchestration_lock_{uuid.uuid4().hex}"
    try:
        lock_path = root / "lock"
        acquired_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        recovered_at = acquired_at + timedelta(seconds=PRODUCTION_RUN_LOCK_STALE_AGE_SECONDS + 5)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({"pid": 111, "acquired_at_utc": acquired_at.isoformat(), "owner_id": "abandoned", "hostname": "worker-a"}), encoding="utf-8")
        lock = ProductionRunLock(lock_path, owner_id="retry", now_provider=lambda: recovered_at)
        calls = {"count": 0}

        def fake_write():
            calls["count"] += 1
            if calls["count"] == 1:
                raise FileExistsError
            lock_path.write_text(json.dumps({"pid": 222, "acquired_at_utc": recovered_at.isoformat(), "owner_id": "winner", "hostname": "worker-b"}), encoding="utf-8")
            raise FileExistsError

        monkeypatch.setattr(lock, "_write_lockfile", fake_write)
        with pytest.raises(RuntimeError, match="winner"):
            lock.acquire()
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["owner_id"] == "winner"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_manifest_is_completed_only_after_outputs_and_never_trains(monkeypatch):
    root, panel, artifact = make_environment()
    try:
        local = root / "panel.parquet"; panel.to_parquet(local, index=False)
        result = run_daily_pipeline(artifact_dir=artifact, target_panel_path=local, forecasts_root=root / "forecasts", monitoring_root=root / "monitoring", operations_root=root / "operations", skip_source_refresh=True)
        manifest = json.loads((result["run_dir"] / "run_manifest.json").read_text())
        assert manifest["overall_status"] == "completed"
        assert manifest["artifact"]["artifact_run_id"]
        assert manifest["target_panel"]["sha256"]
        assert manifest["monitoring"]["path"]
        assert not any("train" in item["phase"].lower() for item in manifest["phase_statuses"])
    finally: shutil.rmtree(root, ignore_errors=True)


def test_production_orchestration_uses_default_full_refresh():
    root, panel, artifact = make_environment()
    calls = []
    try:
        def capture_refresh(**kwargs):
            calls.append(kwargs)
            return {
                "target_panel": panel,
                "summary": {
                    "refresh_method": "captured",
                    "refresh_strategy": "captured",
                    "source_identifier": "test",
                    "source_dataset_id": "33kz-ixgy",
                    "latest_source_date": panel["target_date"].max().date().isoformat(),
                    "seattle_today": "2024-03-20",
                    "latest_allowed_complete_date": "2024-03-19",
                    "selected_complete_through_date": panel["target_date"].max().date().isoformat(),
                    "target_panel_path": str(root / "panel.parquet"),
                    "target_panel_sha256": target_panel_sha256(panel),
                },
            }

        run_daily_pipeline(
            artifact_dir=artifact,
            target_panel_path=root / "panel.parquet",
            forecasts_root=root / "forecasts",
            monitoring_root=root / "monitoring",
            operations_root=root / "operations",
            refresh_function=capture_refresh,
        )
        assert len(calls) == 1
        assert sorted(calls[0]) == ["expected_neighborhoods", "target_panel_path"]
    finally:
        shutil.rmtree(root, ignore_errors=True)
