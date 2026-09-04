from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go

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


def _stub_crime_context() -> dict:
    return {
        "valid_time": pd.DataFrame(
            {
                "offense_date": [
                    "2026-09-01",
                    "2026-09-02",
                ]
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
):
    monkeypatch.setattr(
        app_module,
        "load_calls_dashboard_context",
        lambda: _stub_calls_context(),
    )
    monkeypatch.setattr(
        app_module,
        "load_crime_dashboard_context",
        lambda: _stub_crime_context(),
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
    def fake_make_crime_daily_figure(context, selected_bins):
        if crime_daily_capture is not None:
            crime_daily_capture["selected_bins"] = selected_bins
        return go.Figure()

    def fake_make_crime_map_figure(
        context,
        selected_bins,
        point_start_date,
        point_end_date,
        show_colorbar,
        point_filters=None,
    ):
        if crime_map_capture is not None:
            crime_map_capture["selected_bins"] = selected_bins
            crime_map_capture["point_start_date"] = point_start_date
            crime_map_capture["point_end_date"] = point_end_date
            crime_map_capture["show_colorbar"] = show_colorbar
            crime_map_capture["point_filters"] = point_filters

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
        "violent||property",
        {"start": "2026-09-01", "end": "2026-09-02"},
        ["map_colorbar"],
        ["theft"],
        ["downtown"],
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
        "point_filters": {
            "offense_sub_categories": ["theft"],
            "mcpp_neighborhoods": ["downtown"],
            "text": "report",
        },
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
        "violent||property",
        {"start": "2026-09-01", "end": "2026-09-02"},
        ["daily"],
        [],
        [],
        "",
    )

    assert overlay_class == "fullscreen-overlay"
    assert title == "Daily crime events"
    assert figure.layout.showlegend is True
    assert capture == {"selected_bins": ["violent", "property"]}
