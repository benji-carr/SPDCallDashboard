"""Refresh and validate the bounded production target panel."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from dashboard.spd_service import load_spd_call_dataset
from forecasting.features.xgboost import prepare_target_panel, validate_daily_panel
from forecasting.paths import TARGET_PANEL_5Y_PATH


SOURCE_DATASET_ID = "33kz-ixgy"
SOURCE_IDENTIFIER = "Seattle Open Data SPD Calls"
SOURCE_PATH = "https://data.seattle.gov/resource/33kz-ixgy.json"
SOURCE_COLUMNS = {"cad_event_number", "cad_event_original_time_queued", "dispatch_neighborhood"}
SEATTLE_TZ = ZoneInfo("America/Los_Angeles")
MIN_FEATURE_HISTORY_DAYS = 28


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seattle_today(now: datetime | None = None) -> pd.Timestamp:
    value = now or datetime.now(SEATTLE_TZ)
    if value.tzinfo is None:
        value = value.replace(tzinfo=SEATTLE_TZ)
    return pd.Timestamp(value.astimezone(SEATTLE_TZ).date())


def validate_source_schema(source: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(SOURCE_COLUMNS - set(source.columns))
    if missing:
        raise ValueError(f"SPD source schema is missing required columns: {missing}")
    frame = source.copy()
    frame["cad_event_number"] = frame["cad_event_number"].astype("string").str.strip()
    frame["dispatch_neighborhood"] = frame["dispatch_neighborhood"].astype("string").str.strip()
    timestamps = pd.to_datetime(frame["cad_event_original_time_queued"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError("SPD source contains invalid queued timestamps.")
    frame["_event_time_local"] = timestamps.dt.tz_convert(SEATTLE_TZ)
    frame = frame.loc[
        frame["cad_event_number"].notna() & frame["cad_event_number"].ne("")
        & frame["dispatch_neighborhood"].notna() & frame["dispatch_neighborhood"].ne("")
        & frame["dispatch_neighborhood"].ne("NULL")
    ].copy()
    if frame.empty:
        raise ValueError("SPD source has no valid CAD events with neighborhoods.")
    return frame


def complete_through_date(source: pd.DataFrame, now: datetime | None = None) -> dict:
    validated = validate_source_schema(source)
    latest_source_date = validated["_event_time_local"].dt.normalize().max().tz_localize(None)
    today = seattle_today(now)
    allowed = today - pd.Timedelta(days=1)
    return {
        "latest_source_date": latest_source_date,
        "seattle_today": today,
        "latest_allowed_complete_date": allowed,
        "selected_complete_through_date": min(latest_source_date, allowed),
    }


def build_target_panel(
    source: pd.DataFrame,
    *,
    expected_neighborhoods: list[str],
    selected_complete_through_date: str | pd.Timestamp,
    rolling_years: int = 5,
) -> pd.DataFrame:
    if not expected_neighborhoods:
        raise ValueError("Expected neighborhood set is empty.")
    frame = validate_source_schema(source)
    selected = pd.Timestamp(selected_complete_through_date).normalize()
    start = (selected - pd.DateOffset(years=rolling_years)) + pd.Timedelta(days=1)
    frame["target_date"] = frame["_event_time_local"].dt.normalize().dt.tz_localize(None)
    frame = frame.loc[(frame["target_date"] >= start) & (frame["target_date"] <= selected)].copy()
    # A CAD event is the target unit.  Duplicate source rows never inflate calls.
    frame = frame.drop_duplicates(["target_date", "dispatch_neighborhood", "cad_event_number"], keep="last")
    expected = sorted(str(value) for value in expected_neighborhoods)
    unexpected = sorted(set(frame["dispatch_neighborhood"].astype(str)) - set(expected))
    if unexpected:
        raise ValueError(f"Source contains neighborhoods incompatible with artifact: {unexpected}")
    dates = pd.date_range(start, selected, freq="D")
    grid = pd.MultiIndex.from_product([dates, expected], names=["target_date", "neighborhood"]).to_frame(index=False)
    counts = (frame.groupby(["target_date", "dispatch_neighborhood"])["cad_event_number"].nunique()
              .rename("calls").reset_index().rename(columns={"dispatch_neighborhood": "neighborhood"}))
    panel = grid.merge(counts, on=["target_date", "neighborhood"], how="left")
    panel["calls"] = panel["calls"].fillna(0.0).astype(float)
    return prepare_target_panel(panel)


def target_panel_sha256(panel: pd.DataFrame) -> str:
    canonical = prepare_target_panel(panel)[["target_date", "neighborhood", "calls"]].copy()
    canonical["target_date"] = canonical["target_date"].dt.strftime("%Y-%m-%d")
    canonical["neighborhood"] = canonical["neighborhood"].astype(str)
    canonical = canonical.sort_values(["target_date", "neighborhood"]).reset_index(drop=True)
    digest = hashlib.sha256("target_date|neighborhood|calls".encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(canonical, index=False).to_numpy(dtype="uint64").tobytes())
    return digest.hexdigest()


def validate_target_panel_for_artifact(panel: pd.DataFrame, expected_neighborhoods: list[str], complete_date: str | pd.Timestamp) -> pd.DataFrame:
    validated = prepare_target_panel(panel)
    if validated.empty or not np.isfinite(validated["calls"]).all():
        raise ValueError("Target panel is empty or contains non-finite calls.")
    actual = set(validated["neighborhood"].astype(str))
    expected = set(str(value) for value in expected_neighborhoods)
    if actual != expected:
        raise ValueError(f"Target panel entity set differs from artifact; missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}")
    validate_daily_panel(validated)
    complete = pd.Timestamp(complete_date).normalize()
    if validated["target_date"].max() != complete or (validated["target_date"] > complete).any():
        raise ValueError("Target panel does not end exactly at the selected complete-through date.")
    if validated["target_date"].nunique() <= MIN_FEATURE_HISTORY_DAYS:
        raise ValueError("Target panel lacks the 28 days required for production features.")
    return validated


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def refresh_production_data(
    *, expected_neighborhoods: list[str], target_panel_path: str | Path = TARGET_PANEL_5Y_PATH,
    fetch_source=load_spd_call_dataset, now: datetime | None = None, page_size: int = 5000,
    timeout: float = 60.0,
) -> dict:
    started = time.monotonic()
    start_utc = _utc_now()
    today = seattle_today(now)
    # Bounded full refresh is deliberate: the existing snapshot is one year, while v1 needs five.
    fetch_start = ((today - pd.DateOffset(years=5)) - pd.Timedelta(days=1)).date().isoformat()
    source = fetch_source(start_date=fetch_start, page_size=page_size, max_pages=None, timeout=timeout)
    dates = complete_through_date(source, now)
    panel = build_target_panel(source, expected_neighborhoods=expected_neighborhoods,
                               selected_complete_through_date=dates["selected_complete_through_date"])
    panel = validate_target_panel_for_artifact(panel, expected_neighborhoods, dates["selected_complete_through_date"])
    output = Path(target_panel_path)
    _atomic_parquet(panel, output)
    summary = {
        "refresh_started_at_utc": start_utc, "refresh_completed_at_utc": _utc_now(),
        "refresh_elapsed_seconds": round(time.monotonic() - started, 6),
        "source_identifier": SOURCE_IDENTIFIER, "source_dataset_id": SOURCE_DATASET_ID, "source_path": SOURCE_PATH,
        "refresh_method": "full_bounded_existing_repository_method", "revision_lookback_days": None,
        "latest_source_date": dates["latest_source_date"].date().isoformat(),
        "seattle_today": dates["seattle_today"].date().isoformat(),
        "latest_allowed_complete_date": dates["latest_allowed_complete_date"].date().isoformat(),
        "selected_complete_through_date": dates["selected_complete_through_date"].date().isoformat(),
        "target_panel_start": panel["target_date"].min().date().isoformat(), "target_panel_end": panel["target_date"].max().date().isoformat(),
        "n_target_dates": int(panel["target_date"].nunique()), "n_neighborhoods": int(panel["neighborhood"].nunique()), "n_target_rows": int(len(panel)),
        "target_calls": {"mean": float(panel.calls.mean()), "std": float(panel.calls.std(ddof=0)), "min": float(panel.calls.min()), "max": float(panel.calls.max()), "sum": float(panel.calls.sum())},
        "target_panel_sha256": target_panel_sha256(panel), "target_panel_path": str(output),
    }
    return {"source": source, "target_panel": panel, "summary": summary}
