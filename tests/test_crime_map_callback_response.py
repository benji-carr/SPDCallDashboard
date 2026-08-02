from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app import dashboard


ORIGINAL_CRIME_MAP_CALLBACK_KEY = (
    "..crime-map-figure.figure..."
    "crime-map-point-window-label.children.."
)

REMOUNT_CRIME_MAP_CALLBACK_KEY = (
    "..crime-map-graph-container.children..."
    "crime-map-point-window-label.children.."
)


def get_crime_map_callback_key():
    if ORIGINAL_CRIME_MAP_CALLBACK_KEY in dashboard.callback_map:
        return ORIGINAL_CRIME_MAP_CALLBACK_KEY

    if REMOUNT_CRIME_MAP_CALLBACK_KEY in dashboard.callback_map:
        return REMOUNT_CRIME_MAP_CALLBACK_KEY

    raise AssertionError(
        "Could not find a crime map callback key in dashboard.callback_map"
    )


def get_outputs_for_callback_key(callback_key):
    if callback_key == ORIGINAL_CRIME_MAP_CALLBACK_KEY:
        return [
            {
                "id": "crime-map-figure",
                "property": "figure",
            },
            {
                "id": "crime-map-point-window-label",
                "property": "children",
            },
        ]

    if callback_key == REMOUNT_CRIME_MAP_CALLBACK_KEY:
        return [
            {
                "id": "crime-map-graph-container",
                "property": "children",
            },
            {
                "id": "crime-map-point-window-label",
                "property": "children",
            },
        ]

    raise AssertionError(f"Unknown callback key: {callback_key}")


def request_crime_map_callback(
    selected_category_value,
    start_date="2026-07-01",
    end_date="2026-07-30",
    legend_values=None,
):
    if legend_values is None:
        legend_values = []

    callback_key = get_crime_map_callback_key()
    outputs = get_outputs_for_callback_key(callback_key)

    payload = {
        "output": callback_key,
        "outputs": outputs,
        "inputs": [
            {
                "id": "crime-category-filter",
                "property": "value",
                "value": selected_category_value,
            },
            {
                "id": "crime-daily-visible-range-store",
                "property": "data",
                "value": {
                    "start": start_date,
                    "end": end_date,
                },
            },
            {
                "id": "crime-legend-toggle",
                "property": "value",
                "value": legend_values,
            },
        ],
        "state": [],
        "changedPropIds": [
            "crime-category-filter.value",
        ],
    }

    client = dashboard.server.test_client()

    response = client.post(
        "/_dash-update-component",
        json=payload,
    )

    assert response.status_code == 200, response.get_data(as_text=True)

    return response.get_json()


def extract_figure_from_callback_response(response_json):
    response_data = response_json["response"]

    if "crime-map-figure" in response_data:
        return response_data["crime-map-figure"]["figure"]

    if "crime-map-graph-container" in response_data:
        graph_component = response_data["crime-map-graph-container"]["children"]
        return graph_component["props"]["figure"]

    raise AssertionError(
        f"Could not find crime map figure in response: {response_data.keys()}"
    )


def get_point_trace_counts(figure):
    counts = {}

    for trace in figure["data"]:
        trace_type = str(trace.get("type", "")).lower()

        if trace_type not in ["scattermapbox", "scattermap"]:
            continue

        trace_name = trace.get("name", "unnamed")
        lat_values = trace.get("lat", [])

        counts[trace_name] = len(lat_values)

    return counts


def get_choropleth_trace(figure):
    for trace in figure["data"]:
        trace_type = str(trace.get("type", "")).lower()

        if trace_type in ["choroplethmapbox", "choroplethmap"]:
            return trace

    raise AssertionError("No choropleth map trace found")


def test_crime_map_callback_returns_property_only_when_property_selected():
    response_json = request_crime_map_callback(
        selected_category_value="property crime",
    )

    figure = extract_figure_from_callback_response(response_json)

    counts = get_point_trace_counts(figure)

    nonzero_categories = {
        category
        for category, count in counts.items()
        if count > 0
    }

    assert nonzero_categories <= {"property crime"}


def test_crime_map_callback_returns_violent_only_when_violent_selected():
    response_json = request_crime_map_callback(
        selected_category_value="violent crime",
    )

    figure = extract_figure_from_callback_response(response_json)

    counts = get_point_trace_counts(figure)

    nonzero_categories = {
        category
        for category, count in counts.items()
        if count > 0
    }

    assert nonzero_categories <= {"violent crime"}


def test_crime_map_callback_switching_does_not_return_old_point_categories():
    first_response = request_crime_map_callback(
        selected_category_value="property crime",
    )

    second_response = request_crime_map_callback(
        selected_category_value="violent crime",
    )

    first_figure = extract_figure_from_callback_response(first_response)
    second_figure = extract_figure_from_callback_response(second_response)

    first_counts = get_point_trace_counts(first_figure)
    second_counts = get_point_trace_counts(second_figure)

    first_nonzero = {
        category
        for category, count in first_counts.items()
        if count > 0
    }

    second_nonzero = {
        category
        for category, count in second_counts.items()
        if count > 0
    }

    assert first_nonzero <= {"property crime"}
    assert second_nonzero <= {"violent crime"}


def test_crime_map_callback_colorbar_toggle_changes_server_response():
    no_colorbar_response = request_crime_map_callback(
        selected_category_value="property crime",
        legend_values=[],
    )

    colorbar_response = request_crime_map_callback(
        selected_category_value="property crime",
        legend_values=["map_colorbar"],
    )

    no_colorbar_figure = extract_figure_from_callback_response(no_colorbar_response)
    colorbar_figure = extract_figure_from_callback_response(colorbar_response)

    no_colorbar_choropleth = get_choropleth_trace(no_colorbar_figure)
    colorbar_choropleth = get_choropleth_trace(colorbar_figure)

    assert no_colorbar_choropleth.get("showscale") is False
    assert colorbar_choropleth.get("showscale") is True


if __name__ == "__main__":
    import pytest

    raise SystemExit(
        pytest.main(
            [
                __file__,
                "-vv",
                "-s",
                "--tb=long",
            ]
        )
    )