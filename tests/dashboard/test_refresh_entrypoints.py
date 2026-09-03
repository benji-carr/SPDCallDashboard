from scripts.dashboard import refresh_crime_data, refresh_spd_data


def test_spd_refresh_command_verifies_freshness(monkeypatch):
    calls = []
    monkeypatch.setattr(refresh_spd_data, "incremental_refresh_spd_call_snapshot", lambda: calls.append("refresh"))
    monkeypatch.setattr(refresh_spd_data, "check_spd_calls_freshness", lambda: calls.append("check"))
    refresh_spd_data.main()
    assert calls == ["refresh", "check"]


def test_crime_refresh_command_verifies_freshness(monkeypatch):
    calls = []
    monkeypatch.setattr(refresh_crime_data, "incremental_refresh_crime_snapshot", lambda: calls.append("refresh"))
    monkeypatch.setattr(refresh_crime_data, "check_crime_freshness", lambda: calls.append("check"))
    refresh_crime_data.main()
    assert calls == ["refresh", "check"]
