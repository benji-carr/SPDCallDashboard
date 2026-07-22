from spd_config import TARGET_IMPORTANCE_BINS
from spd_dashboard_data import load_dashboard_context
from spd_dashboard_figures import (
    make_daily_figure,
    make_map_figure,
    make_volume_response_scatter,
)


def main():
    print("Loading dashboard context...")
    context = load_dashboard_context()

    print("Context keys:")
    print(sorted(context.keys()))

    print("Building daily figure...")
    daily_fig = make_daily_figure(
        context=context,
        selected_bins=TARGET_IMPORTANCE_BINS,
    )

    print(f"Daily figure traces: {len(daily_fig.data)}")

    print("Building scatter figure...")
    scatter_fig = make_volume_response_scatter(
        context=context,
        selected_bins=TARGET_IMPORTANCE_BINS,
    )

    print(f"Scatter figure traces: {len(scatter_fig.data)}")

    print("Building map figure...")
    map_fig = make_map_figure(
        context=context,
        selected_bins=TARGET_IMPORTANCE_BINS,
        show_colorbar=False,
    )

    print(f"Map figure traces: {len(map_fig.data)}")

    if len(daily_fig.data) == 0:
        raise ValueError("Daily figure has no traces")

    if len(scatter_fig.data) == 0:
        raise ValueError("Scatter figure has no traces")

    if len(map_fig.data) == 0:
        raise ValueError("Map figure has no traces")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()