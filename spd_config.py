from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

GEO_PROCESSED_DIR = DATA_PROCESSED_DIR / "geography"
GEO_EXTERNAL_DIR = DATA_EXTERNAL_DIR / "boundaries"

EVENT_ID_COLUMN = "cad_event_number"
ROW_ID_COLUMN = "call_sign_dispatch_id"
TIME_COLUMN = "cad_event_original_time_queued"
ARRIVAL_TIME_COLUMN = "cad_event_arrived_time"

LAT_COL = "dispatch_latitude"
LON_COL = "dispatch_longitude"

PLOTLY_TEMPLATE = "plotly_dark"
PLOT_BG = "#545455"
PAPER_BG = "#111111"

PLOTLY_SEATTLE_CENTER = {
    "lat": 47.6062,
    "lon": -122.3321,
}

PLOTLY_MAP_STYLE = "carto-darkmatter"

TARGET_IMPORTANCE_BINS = [
    "property/nonviolent",
    "drug-related",
    "violent/person crime",
]

IMPORTANT_EVENT_GROUPS = [
    "assault",
    "burglary",
    "domestic disturbance/violence",
    "kidnap",
    "rape",
    "robbery",
    "sex offenses (non-rape)",
    "theft",
    "narcotics",
    "homicide",
]

MCPP_GEOJSON_URL = (
    "https://services.arcgis.com/ZOyb2t4B0UYuYNYH/ArcGIS/rest/services/"
    "SPD_Boundaries/FeatureServer/0/query"
    "?where=1%3D1"
    "&outFields=*"
    "&outSR=4326"
    "&f=geojson"
)

POPULATION_PATH = DATA_EXTERNAL_DIR / "neighborhood_population.csv"