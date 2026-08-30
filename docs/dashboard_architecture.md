## Application Layer:

The application layer is primarily controlled by `app.py`. This script takes the contexts and figure building functions from the previous layers and connects them to the Dash user interface. This means that `app.py` does relatively little data processing itself; its main responsibilities are page layouts, routing, storing user selections, and determining when figures need to be rebuilt.

### create_app():

`create_app()` initializes the Dash app, loads the calls and crime dashboard contexts, creates the options used by dropdowns, finds the default map date ranges, and defines the cached figure functions, page layouts, and callbacks used by the app. Both dashboard contexts are loaded once when the application is created rather than being rebuilt every time a callback runs.

Several figure building functions inside `create_app()` use `lru_cache`. This allows previously generated figures to be reused when the same combination of category, date range, and legend settings is requested again rather than performing the same work repeatedly.

### Page Layouts:

There are three main page-building functions:

* `make_landing_page()` builds the home page and links users to the two dashboards.
* `make_calls_page()` builds the calls dashboard and its map, time series, scatter plot, controls, and fullscreen elements.
* `make_crime_page()` builds the crime dashboard and its map, time series, and controls.

The root layout contains a `dcc.Location` component and a `page-content` container. The `display_page()` callback reads the URL and decides which page-building function should populate the container. `/crime` opens the crime dashboard, `/calls` opens the calls dashboard, and other paths return the landing page.

### Date Range Stores:

Both dashboards use a `dcc.Store` to keep track of the visible range of the daily time series. When the user zooms, pans, or uses the range slider, the `relayoutData` from the Plotly figure is passed through `extract_daily_visible_date_range()` and stored. The map callback then reads the same stored dates and uses them as the point start and end dates. This is what links the time series controls to the map points.

### Figure Update Callbacks:

The calls dashboard has callbacks for the daily figure, scatter plot, and map. Changing the top-level type-of-crime dropdown causes the corresponding figures to be rebuilt using the new selected bins. The map additionally responds to the stored date range and whether the map color scale is enabled.

The crime dashboard follows the same pattern for the daily figure and map. The crime map has additional point-only filters for offense sub-category, neighborhood, and text searches. These filters are passed to the visualization layer and do not change the neighborhood choropleth totals.

The calls dashboard also contains callbacks for the fullscreen functionality. The selected figure is stored in `fullscreen-figure-store`, then the corresponding cached map, daily figure, or scatter plot is displayed in the fullscreen overlay.

## Automated Refresh Pipeline:

The rolling snapshot layer described earlier is run automatically through `.github/workflows/daily_spd_refresh.yml`. The workflow can be manually triggered but is also scheduled to run once every day.

The workflow performs the following steps:

```text
Check out repository
        |
        ▼
Set up Python and install requirements
        |
        ▼
Refresh SPD calls snapshot
        |
        ▼
Refresh crime snapshot
        |
        ▼
Remove old geography lookup caches
        |
        ▼
Rebuild both dashboard contexts
        |
        ▼
Run dashboard smoke test
        |
        ▼
Commit refreshed data files
        |
        ▼
Push changes to the repository
```

The geography lookup files are removed before rebuilding the dashboard contexts so the lookup between events and MCPP neighborhood polygons is recreated using the newest snapshot data.

After the contexts are rebuilt the workflow runs `scripts/dashboard/smoke_check.py`. The smoke check loads the calls dashboard context and attempts to build the daily figure, scatter plot, and map. The check fails if any of the figures contain no traces.

If all of these steps succeed, the updated parquet files, metadata files, and geographic lookup files are committed by `github-actions[bot]` and pushed back into the repository.

## Calls Dashboard vs. Crime Dashboard Architecture:

The two dashboards intentionally follow nearly the same architecture. Each has a query module, client module, data cleaning module, service module, snapshot module, dashboard data module, and dashboard figures module. This makes changes to one dashboard easier to understand using the structure of the other.

The biggest differences occur after the source data has been loaded. The calls dashboard has dispatch records associated with CAD events, so multiple rows can belong to one event. It also has queued and arrival timestamps, allowing response times to be calculated. These response times are used in the map hover data and the volume versus response-time scatter plot.

The crime dashboard does not have response-time information, but it has the additional problem of offenses that have a neighborhood listed while their exact coordinates are unavailable. For this reason the crime dashboard separates mappable and unmappable offenses and combines their unique event counts when calculating neighborhood totals. Only mappable offenses can be displayed as map points.

The crime dashboard also has additional point filters for sub-category, neighborhood, and text searches. The calls dashboard currently uses only the broad top-level type-of-crime selection and date window for its event points.

## Dashboard File Reference:

The following is a basic reference for where changes to different parts of the dashboard should be made:

```text
Change API query fields or SoQL:
    dashboard/*_query.py

Change API endpoint/request behavior:
    dashboard/*_client.py

Change initial API data cleaning:
    dashboard/*_data.py

Change pagination or full dataset loading:
    dashboard/*_service.py

Change snapshot reading/writing:
    dashboard/*_snapshot.py

Change daily refresh behavior:
    scripts/dashboard/refresh_*_data.py

Change dashboard-specific data preparation:
    dashboard/*_dashboard_data.py

Change plots, hover data, or figure-specific filters:
    dashboard/*_dashboard_figures.py

Change calls event category definitions:
    dashboard/spd_event_bins.py

Change shared calls configuration, paths, or map settings:
    dashboard/spd_config.py

Change page layouts, controls, callbacks, or plot interactions:
    app.py

Change dashboard styling:
    assets/style.css

Change automated daily refresh behavior:
    .github/workflows/daily_spd_refresh.yml

Change dashboard smoke testing:
    scripts/dashboard/smoke_check.py
```
