import pandas as pd
import pytest

from dashboard.crime_client import fetch_latest_crime_dashboard_record
from dashboard.spd_client import fetch_latest_spd_dashboard_record
from scripts.dashboard.check_data_freshness import (
    assert_fresh,
    latest_dashboard_date,
    latest_source_date,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, endpoint, *, params, timeout):
        self.calls.append({"endpoint": endpoint, "params": params, "timeout": timeout})
        return FakeResponse(self.payload)


def test_spd_source_query_requests_newest_valid_dashboard_record():
    session = FakeSession([{"cad_event_original_time_queued": "2026-09-03T12:00:00.000", "cad_event_number": "123"}])
    record = fetch_latest_spd_dashboard_record(session=session)
    assert record["cad_event_number"] == "123"
    assert session.calls[0]["params"] == {
        "$select": "cad_event_original_time_queued,cad_event_number",
        "$where": "cad_event_original_time_queued IS NOT NULL AND cad_event_number IS NOT NULL",
        "$order": "cad_event_original_time_queued DESC",
        "$limit": 1,
    }


def test_crime_source_query_requests_newest_valid_dashboard_record(monkeypatch):
    calls = []

    def fake_get(endpoint, *, params, timeout):
        calls.append({"endpoint": endpoint, "params": params, "timeout": timeout})
        return FakeResponse([{"offense_date": "2026-09-03T12:00:00.000", "offense_id": "456"}])

    monkeypatch.setattr("dashboard.crime_client.requests.get", fake_get)
    assert fetch_latest_crime_dashboard_record()["offense_id"] == "456"
    assert calls[0]["params"]["$order"] == "offense_date DESC, offense_id ASC"
    assert "offense_date IS NOT NULL" in calls[0]["params"]["$where"]
    assert "offense_id IS NOT NULL" in calls[0]["params"]["$where"]


@pytest.mark.parametrize("record", [[], [{}], [{"cad_event_number": "1"}], [{"cad_event_original_time_queued": "not-a-date", "cad_event_number": "1"}]])
def test_malformed_or_empty_source_records_fail_clearly(record):
    if record == []:
        with pytest.raises(ValueError, match="no valid dashboard records"):
            fetch_latest_spd_dashboard_record(session=FakeSession(record))
    else:
        with pytest.raises(ValueError, match="missing|invalid"):
            latest_source_date(record[0], time_column="cad_event_original_time_queued", event_id_column="cad_event_number", label="SPD calls")


def test_latest_dashboard_date_uses_dashboard_validity_rules():
    frame = pd.DataFrame(
        {
            "cad_event_original_time_queued": ["2026-09-02T10:00:00", "invalid", "2026-09-04T10:00:00"],
            "cad_event_number": ["1", "2", None],
        }
    )
    assert latest_dashboard_date(frame, time_column="cad_event_original_time_queued", event_id_column="cad_event_number", label="SPD calls") == pd.Timestamp("2026-09-02")


def test_equal_calendar_dates_pass_despite_time_of_day():
    assert_fresh(label="Crime", source_date=pd.Timestamp("2026-09-03T23:59:59"), dashboard_date=pd.Timestamp("2026-09-03T00:00:00"))


def test_older_dashboard_date_fails_with_diagnostic_message():
    with pytest.raises(ValueError, match="SPD calls dashboard data is stale: Seattle Open Data latest day=2026-09-03; dashboard latest day=2026-09-02"):
        assert_fresh(label="SPD calls", source_date=pd.Timestamp("2026-09-03"), dashboard_date=pd.Timestamp("2026-09-02"))
