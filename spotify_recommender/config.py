from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
RAW_MUSIC_DATA = RAW_DATA_DIR / "Music_Info.csv"
INTERIM_MUSIC_DATA = INTERIM_DATA_DIR / "Music_Info.csv"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# For data cleaning
DATA_CLEANING_DROP_DUPLICATES_SUBSET: list = [
    "spotify_preview_url",
    "spotify_id",
    "artist",
    "year",
    "duration_ms",
    "tempo",
]
DATA_CLEANING_COLS_TO_DROP: list = ["genre", "spotify_id"]
DATA_CLEANING_FILL_NA_VALS_DICT: dict = {"tags": "no_tags"}
DATA_CLEANING_COLS_TO_LOWERCASE: list = ["name", "artist", "tags"]
