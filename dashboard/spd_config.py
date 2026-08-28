import os
from pathlib import Path
import logging

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

CARTO_BASEMAP_API_KEY = os.getenv(
    "CARTO_BASEMAP_API_KEY",
    "",
).strip()

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

if CARTO_BASEMAP_API_KEY:
    PLOTLY_MAP_STYLE = (
        "https://basemaps.cartocdn.com/gl/"
        "dark-matter-gl-style/style.json"
        f"?key={CARTO_BASEMAP_API_KEY}"
    )
    logger.info("CARTO_BASEMAP_API_KEY was successfully imported")
else:
    logger.warning(
        "CARTO_BASEMAP_API_KEY was not found in the environment. "
        "Falling back to the OpenStreetMap basemap."
    )
    PLOTLY_MAP_STYLE = "open-street-map"

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
