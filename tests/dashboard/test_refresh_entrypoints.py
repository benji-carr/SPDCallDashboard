from scripts.dashboard import refresh_crime_data, refresh_spd_data


def test_spd_refresh_command_verifies_freshness(monkeypatch):
    calls = []
    fetch_sources = []

    def check_freshness(*, fetch_source):
        calls.append("check")
        fetch_sources.append(fetch_source)

    monkeypatch.setattr(refresh_spd_data, "incremental_refresh_spd_call_snapshot", lambda: calls.append("refresh"))
    monkeypatch.setattr(refresh_spd_data, "check_spd_calls_freshness", check_freshness)
    refresh_spd_data.main()
    assert calls == ["refresh", "check"]
    assert callable(fetch_sources[0])


def test_crime_refresh_command_verifies_freshness(monkeypatch):
    calls = []
    fetch_sources = []

    def check_freshness(*, fetch_source):
        calls.append("check")
        fetch_sources.append(fetch_source)

    monkeypatch.setattr(refresh_crime_data, "incremental_refresh_crime_snapshot", lambda: calls.append("refresh"))
    monkeypatch.setattr(refresh_crime_data, "check_crime_freshness", check_freshness)
    refresh_crime_data.main()
    assert calls == ["refresh", "check"]
    assert callable(fetch_sources[0])
