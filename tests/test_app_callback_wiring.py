
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import dashboard

def test_crime_map_callback_is_wired_to_crime_controls_only():
    callback_key = (
        "..crime-map-figure.figure..."
        "crime-map-point-window-label.children.."
    )

    assert callback_key in dashboard.callback_map

    callback = dashboard.callback_map[callback_key]

    input_ids = {
        item["id"]
        for item in callback["inputs"]
    }

    assert input_ids == {
        "crime-category-filter",
        "crime-daily-visible-range-store",
        "crime-legend-toggle",
    }


def test_calls_and_crime_map_callbacks_are_separate():
    crime_callback_key = (
        "..crime-map-figure.figure..."
        "crime-map-point-window-label.children.."
    )

    calls_callback_key = (
        "..map-figure.figure..."
        "map-point-window-label.children.."
    )

    assert crime_callback_key in dashboard.callback_map
    assert calls_callback_key in dashboard.callback_map

    crime_inputs = {
        item["id"]
        for item in dashboard.callback_map[crime_callback_key]["inputs"]
    }

    calls_inputs = {
        item["id"]
        for item in dashboard.callback_map[calls_callback_key]["inputs"]
    }

    assert "crime-category-filter" in crime_inputs
    assert "importance-bin-filter" in calls_inputs
    assert "importance-bin-filter" not in crime_inputs
    assert "crime-category-filter" not in calls_inputs

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