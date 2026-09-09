from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import pytest

import app as app_module


CALLBACK_KEY = (
    "..crime-fullscreen-overlay.className..."
    "crime-fullscreen-title.children..."
    "crime-fullscreen-figure.figure.."
)


def _stub_calls_context() -> dict:
    return {
        "valid_time": pd.DataFrame(
            {
                "cad_event_original_time_queued": [
                    "2026-09-01",
                    "2026-09-02",
                ],
                "cad_event_number": ["call-1", "call-2"],
            }
        )
    }


def _stub_crime_context(start="2026-09-01") -> dict:
    return {
        "valid_time": pd.DataFrame(
            {
                "offense_date": [start, "2026-09-02"],
                "offense_sub_category": ["theft", "theft"],
                "mcpp_neighborhood": ["downtown", "downtown"],
            }
        ),
        "event_mcpp": pd.DataFrame(
            {
                "offense_sub_category": ["theft"],
                "mcpp_neighborhood": ["downtown"],
            }
        ),
    }


def _build_stub_app(
    monkeypatch,
    *,
    crime_map_capture=None,
    crime_daily_capture=None,
    crime_start="2026-09-01",
):
    monkeypatch.setattr(
        app_module,
        "load_calls_dashboard_context",
        lambda: _stub_calls_context(),
    )
    monkeypatch.setattr(
        app_module,
        "load_crime_dashboard_context",
        lambda: _stub_crime_context(crime_start),
    )
    monkeypatch.setattr(
        app_module,
        "make_calls_daily_figure",
        lambda context, selected_bins: go.Figure(),
    )
    monkeypatch.setattr(
        app_module,
        "make_calls_map_figure",
        lambda context, selected_bins, point_start_date, point_end_date, show_colorbar: go.Figure(),
    )
    monkeypatch.setattr(
        app_module,
        "make_calls_scatter_figure",
        lambda context, selected_bins: go.Figure(),
    )
    def fake_make_crime_daily_figure(context, selected_bins, analysis_state=None):
        if crime_daily_capture is not None:
            crime_daily_capture["selected_bins"] = selected_bins
            crime_daily_capture["analysis_state"] = analysis_state
        return go.Figure()

    def fake_make_crime_map_figure(
        context,
        selected_bins,
        point_start_date,
        point_end_date,
        show_colorbar,
        point_filters=None,
        analysis_state=None,
    ):
        if crime_map_capture is not None:
            crime_map_capture["selected_bins"] = selected_bins
            crime_map_capture["point_start_date"] = point_start_date
            crime_map_capture["point_end_date"] = point_end_date
            crime_map_capture["show_colorbar"] = show_colorbar
            crime_map_capture["point_filters"] = point_filters
            crime_map_capture["analysis_state"] = analysis_state

        fig = go.Figure()
        fig.add_trace(
            go.Scattermap(
                lat=[47.6, 47.61],
                lon=[-122.33, -122.34],
            )
        )
        return fig

    monkeypatch.setattr(
        app_module,
        "make_crime_daily_figure",
        fake_make_crime_daily_figure,
    )
    monkeypatch.setattr(
        app_module,
        "make_crime_map_figure",
        fake_make_crime_map_figure,
    )

    return app_module.create_app()


def _collect_component_ids(component):
    ids = set()
    component_id = getattr(component, "id", None)

    if component_id is not None:
        ids.add(str(component_id))

    children = getattr(component, "children", None)

    if children is None:
        return ids

    if isinstance(children, (list, tuple)):
        for child in children:
            ids.update(_collect_component_ids(child))
        return ids

    ids.update(_collect_component_ids(children))
    return ids


def test_crime_layout_includes_fullscreen_components(monkeypatch):
    app = _build_stub_app(monkeypatch)

    page_callback = app.callback_map["page-content.children"]["callback"].__wrapped__
    crime_page = page_callback("/crime")
    component_ids = _collect_component_ids(crime_page)

    assert "crime-expand-map-button" in component_ids
    assert "crime-expand-daily-button" in component_ids
    assert "crime-fullscreen-figure-store" in component_ids
    assert "crime-fullscreen-overlay" in component_ids
    assert "crime-fullscreen-title" in component_ids
    assert "crime-close-fullscreen-button" in component_ids
    assert "crime-fullscreen-figure" in component_ids


def test_crime_fullscreen_store_callback_switches_targets(monkeypatch):
    app = _build_stub_app(monkeypatch)
    callback = app.callback_map["crime-fullscreen-figure-store.data"]["callback"].__wrapped__

    monkeypatch.setattr(
        app_module,
        "ctx",
        SimpleNamespace(triggered_id="crime-expand-map-button"),
    )
    assert callback(1, None, None) == "map"

    monkeypatch.setattr(
        app_module,
        "ctx",
        SimpleNamespace(triggered_id="crime-expand-daily-button"),
    )
    assert callback(None, 1, None) == "daily"

    monkeypatch.setattr(
        app_module,
        "ctx",
        SimpleNamespace(triggered_id="crime-close-fullscreen-button"),
    )
    assert callback(None, None, 1) is None


def test_crime_fullscreen_overlay_rebuilds_map_with_current_filters(monkeypatch):
    capture = {}
    app = _build_stub_app(
        monkeypatch,
        crime_map_capture=capture,
    )
    callback = app.callback_map[CALLBACK_KEY]["callback"].__wrapped__

    overlay_class, title, figure = callback(
        "map",
        _analysis_state(subcategories=["theft"], neighborhoods=["downtown"]),
        ["map_colorbar"],
        "report",
    )

    assert overlay_class == "fullscreen-overlay"
    assert title == "Map view | 2026-09-01 to 2026-09-02 | 2 visible points"
    assert figure.data
    assert capture == {
        "selected_bins": ["violent", "property"],
        "point_start_date": "2026-09-01",
        "point_end_date": "2026-09-02",
        "show_colorbar": True,
        "point_filters": {"text": "report"},
        "analysis_state": _analysis_state(subcategories=["theft"], neighborhoods=["downtown"]),
    }


def test_crime_fullscreen_overlay_rebuilds_daily_chart_with_legend_state(monkeypatch):
    capture = {}
    app = _build_stub_app(
        monkeypatch,
        crime_daily_capture=capture,
    )
    callback = app.callback_map[CALLBACK_KEY]["callback"].__wrapped__

    overlay_class, title, figure = callback(
        "daily",
        _analysis_state(),
        ["daily"],
        "",
    )

    assert overlay_class == "fullscreen-overlay"
    assert title == "Daily crime events"
    assert figure.layout.showlegend is True
    assert capture == {"selected_bins": ["violent", "property"], "analysis_state": _analysis_state()}


def _analysis_state(subcategories=None, neighborhoods=None):
    return {
        "start_date": "2026-09-01", "end_date": "2026-09-02",
        "crime_categories": ["violent", "property"],
        "crime_subcategories": subcategories or [], "neighborhoods": neighborhoods or [],
    }


def test_crime_type_control_has_individual_options_and_list_state(monkeypatch):
    app = _build_stub_app(monkeypatch)
    page = app.callback_map["page-content.children"]["callback"].__wrapped__("/crime")

    def find(component, target):
        if getattr(component, "id", None) == target:
            return component
        children = getattr(component, "children", None) or []
        if not isinstance(children, (list, tuple)):
            children = [children]
        for child in children:
            result = find(child, target)
            if result is not None:
                return result

    control = find(page, "crime-category-filter")
    categories = app_module.TARGET_CRIME_CATEGORIES
    assert control.multi is True
    assert control.value == categories
    assert [option["value"] for option in control.options] == categories
    assert len(control.options) == 3
    callback = app.callback_map["crime-analysis-state-store.data"]["callback"].__wrapped__
    for selected in [categories, categories[1:], categories[1:2]]:
        state = callback("2026-09-02", "2026-09-02", selected, [], [])
        assert state["crime_categories"] == selected
        assert state["crime_categories"] is not selected
    assert callback("2026-09-02", "2026-09-02", [], [], [])["crime_categories"] == categories
    assert find(page, "crime-analysis-period-heading").children == "Period of Analysis"
    assert find(page, "crime-analysis-period-duration").children == "(Latest day)"


def test_crime_analysis_controls_and_state_ownership(monkeypatch):
    app = _build_stub_app(monkeypatch)
    page = app.callback_map["page-content.children"]["callback"].__wrapped__("/crime")
    ids = _collect_component_ids(page)
    assert {"crime-analysis-state-store", "crime-analysis-period-label",
            "crime-category-filter", "crime-subcategory-filter",
            "crime-neighborhood-filter", "crime-point-text-filter"} <= ids
    assert "crime-point-subcategory-filter" not in ids
    assert "crime-point-neighborhood-filter" not in ids
    state_inputs = app.callback_map["crime-analysis-state-store.data"]["inputs"]
    assert {item["id"] for item in state_inputs} == {
        "crime-analysis-date-range", "crime-category-filter",
        "crime-subcategory-filter", "crime-neighborhood-filter",
    }
    map_key = next(key for key in app.callback_map if "crime-map-graph-container" in key)
    assert {item["id"] for item in app.callback_map[map_key]["inputs"]} == {
        "crime-analysis-state-store", "crime-legend-toggle", "crime-point-text-filter",
    }
    assert {item["id"] for item in app.callback_map["crime-daily-figure.figure"]["inputs"]} == {
        "crime-analysis-state-store", "crime-legend-toggle",
    }


def test_analysis_state_callback_defaults_and_selections(monkeypatch):
    import json
    app = _build_stub_app(monkeypatch)
    callback = app.callback_map["crime-analysis-state-store.data"]["callback"].__wrapped__
    default = callback("2026-09-02", "2026-09-02", None, None, None)
    assert default == {
        "start_date": "2026-09-02", "end_date": "2026-09-02",
        "crime_categories": app_module.TARGET_CRIME_CATEGORIES,
        "crime_subcategories": [], "neighborhoods": [],
    }
    state = callback("2026-09-01", "2026-09-02",
                     ["violent", "property"], ["theft"], ["downtown", "ballard"])
    assert json.loads(json.dumps(state)) == state
    assert state == _analysis_state(["theft"], ["downtown", "ballard"])
    key = "crime-analysis-period-duration.children"
    assert app.callback_map[key]["callback"].__wrapped__(state) == "(Latest day)"


def test_crime_calendar_presets_survive_range_and_state_callbacks(monkeypatch):
    app = _build_stub_app(monkeypatch, crime_start="2025-01-01")
    range_callback = _picker_callback(app)
    state_callback = app.callback_map["crime-analysis-state-store.data"]["callback"].__wrapped__
    period_callback = app.callback_map[
        "crime-analysis-period-duration.children"
    ]["callback"].__wrapped__
    for start, expected in [("2026-08-26", "1 week"), ("2026-08-02", "1 month"),
                            ("2025-09-02", "1 year")]:
        selected_range = range_callback({"xaxis.range": [start, "2026-09-02"]}, "2026-09-02", "2026-09-02")
        state = state_callback(*selected_range, app_module.TARGET_CRIME_CATEGORIES, [], [])
        assert state["start_date"] == start
        assert state["end_date"] == "2026-09-02"
        assert period_callback(state) == f"(Latest {expected.removeprefix('1 ')})"


def test_map_and_daily_callbacks_use_common_state(monkeypatch):
    map_capture, daily_capture = {}, {}
    app = _build_stub_app(monkeypatch, crime_map_capture=map_capture, crime_daily_capture=daily_capture)
    state = _analysis_state(["theft"], ["downtown"])
    key = next(key for key in app.callback_map if "crime-map-graph-container" in key)
    callback = app.callback_map[key]["callback"].__wrapped__
    graph, label = callback(state, ["map_colorbar"], "report")
    assert map_capture["analysis_state"] == state
    assert map_capture["point_filters"] == {"text": "report"}
    assert map_capture["show_colorbar"] is True
    assert "2" in label
    daily = app.callback_map["crime-daily-figure.figure"]["callback"].__wrapped__
    fig = daily(state, ["daily"])
    assert daily_capture["analysis_state"] == state
    assert list(fig.layout.xaxis.range) == ["2026-09-01", "2026-09-02"]
    assert fig.layout.showlegend is True
    callback(state, [], "another search")
    assert map_capture["analysis_state"] == state
    assert map_capture["show_colorbar"] is False


def _picker_callback(app):
    key = next(key for key in app.callback_map if "crime-analysis-date-range.start_date" in key)
    return app.callback_map[key]["callback"].__wrapped__


def test_crime_date_sync_preserves_slider_and_ignores_presentation(monkeypatch):
    import pytest
    from dash.exceptions import PreventUpdate
    app = _build_stub_app(monkeypatch)
    callback = _picker_callback(app)
    assert callback({"xaxis.range": ["2026-09-01", "2026-09-02"]},
                    "2026-09-02", "2026-09-02") == ("2026-09-01", "2026-09-02")
    with pytest.raises(PreventUpdate):
        callback({"autosize": True}, "2026-09-01", "2026-09-02")
    with pytest.raises(PreventUpdate):
        callback({"xaxis.range": ["2026-09-01", "2026-09-02"]},
                 "2026-09-01", "2026-09-02")


def test_picker_defaults_bounds_and_no_competing_store(monkeypatch):
    app = _build_stub_app(monkeypatch)
    page = app.callback_map["page-content.children"]["callback"].__wrapped__("/crime")
    def walk(node):
        yield node
        children = getattr(node, "children", None)
        if children is None:
            return
        if not isinstance(children, (list, tuple)):
            children = [children]
        for child in children:
            yield from walk(child)
    nodes = {getattr(node, "id", None): node for node in walk(page)}
    picker = nodes["crime-analysis-date-range"]
    assert picker.start_date == picker.end_date == "2026-09-02"
    assert picker.min_date_allowed == "2026-09-01"
    assert picker.max_date_allowed == "2026-09-02"
    assert picker.minimum_nights == 0
    assert "crime-daily-visible-range-store" not in nodes
    assert [(i["id"], i["property"]) for i in app.callback_map["crime-analysis-state-store.data"]["inputs"]][:2] == [
        ("crime-analysis-date-range", "start_date"), ("crime-analysis-date-range", "end_date"),
    ]


def test_invalid_picker_edits_preserve_state_and_chart_clamps_to_data(monkeypatch):
    import pytest
    from dash.exceptions import PreventUpdate
    app = _build_stub_app(monkeypatch)
    state = app.callback_map["crime-analysis-state-store.data"]["callback"].__wrapped__
    for start, end in [(None, "2026-09-02"), ("", "2026-09-02"),
                       ("invalid", "2026-09-02"), ("2026-09-02", "2026-09-01"),
                       ("2026-08-31", "2026-09-02"), ("2026-09-01", "2026-09-03")]:
        with pytest.raises(PreventUpdate):
            state(start, end, None, [], [])
    assert _picker_callback(app)({"xaxis.range": ["2020-01-01", "2030-01-01"]},
                                 "2026-09-02", "2026-09-02") == ("2026-09-01", "2026-09-02")


def test_manual_dates_drive_daily_viewport_and_period_label(monkeypatch):
    app = _build_stub_app(monkeypatch, crime_start="2025-01-01")
    state_callback = app.callback_map["crime-analysis-state-store.data"]["callback"].__wrapped__
    state = state_callback("2026-04-01T12:00:00", "2026-07-31", None, [], [])
    figure = app.callback_map["crime-daily-figure.figure"]["callback"].__wrapped__(state, [])
    assert list(figure.layout.xaxis.range) == ["2026-04-01", "2026-07-31"]
    key = "crime-analysis-period-duration.children"
    duration = app.callback_map[key]["callback"].__wrapped__(state)
    assert "Last" not in duration and "Latest" not in duration
    same_day = state_callback("2026-09-02", "2026-09-02", None, [], [])
    figure = app.callback_map["crime-daily-figure.figure"]["callback"].__wrapped__(same_day, [])
    assert list(figure.layout.xaxis.range) == ["2026-09-02", "2026-09-02 23:59:59.999"]
    assert app.callback_map[key]["callback"].__wrapped__(same_day) == "(Latest day)"


def test_invalid_inline_date_edit_reverts_to_last_valid_state(monkeypatch):
    from dash.exceptions import PreventUpdate

    app = _build_stub_app(monkeypatch)
    callback = _picker_callback(app)
    previous = _analysis_state()
    monkeypatch.setattr(
        app_module,
        "ctx",
        SimpleNamespace(triggered_id="crime-analysis-date-range"),
    )
    assert callback(None, None, "2026-09-02", previous) == (
        "2026-09-01", "2026-09-02",
    )
    assert callback(None, "2026-09-02", "2026-09-01", previous) == (
        "2026-09-01", "2026-09-02",
    )
    with pytest.raises(PreventUpdate):
        callback(None, "2026-09-01", "2026-09-02", previous)
