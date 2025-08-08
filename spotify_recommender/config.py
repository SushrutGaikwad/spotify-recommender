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
RAW_SONGS_DATA = RAW_DATA_DIR / "Music_Info.csv"
CLEANED_SONGS_DATA = INTERIM_DATA_DIR / "Music_Info_cleaned.csv"
CLEANED_SONGS_DATA_CBF = INTERIM_DATA_DIR / "Music_Info_cleaned_CBF.csv"
TRANSFORMED_SONGS_DATA_CBF = PROCESSED_DATA_DIR / "Music_Info_cleaned_transformed_CBF.npz"

MODELS_DIR = PROJ_ROOT / "models"
TRAINED_TRANSFORMER_CBF = MODELS_DIR / "trained_transformer_CBF.joblib"

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

# For content-based filtering data preparation
CONTENT_BASED_FILTERING_DATA_PREP_COLS_TO_DROP: list = ["track_id", "name", "spotify_preview_url"]

# For content-based filtering data transformation
CONTENT_BASED_FILTERING_DATA_TRANS_TFIDF_MAX_FEATURES: int = 85
CONTENT_BASED_FILTERING_DATA_TRANS_FREQUENCY_ENCODE_COLS: list = ["year"]
CONTENT_BASED_FILTERING_DATA_TRANS_OHE_COLS: list = ["artist", "time_signature", "key"]
CONTENT_BASED_FILTERING_DATA_TRANS_TFIDF_COL: str = "tags"
CONTENT_BASED_FILTERING_DATA_TRANS_STD_SCALER_COLS: list = ["duration_ms", "loudness", "tempo"]
CONTENT_BASED_FILTERING_DATA_TRANS_MIN_MAX_SCALER_COLS: list = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]

# For content-based filtering recommendation
CONTENT_BASED_FILTERING_RECO_K: int = 10
CONTENT_BASED_FILTERING_RECO_SONG_NAME: str = "Mockingbird"

# For collaborative filtering

# Paths
USER_HISTORY_DATA = RAW_DATA_DIR / "User_Listening_History.csv"
COLLAB_FILTERED_SONGS_DATA = INTERIM_DATA_DIR / "collab_filtered_songs.csv"
COLLAB_TRACK_IDS_NPY = PROCESSED_DATA_DIR / "collab_track_ids.npy"
COLLAB_INTERACTION_MATRIX_NPZ = PROCESSED_DATA_DIR / "collab_interaction_matrix.npz"

# Parameters
COLLAB_K: int = 10
COLLAB_MIN_PLAYS: float = 0.0  # if you want to drop super-low counts later
COLLAB_FILTERING_SONG_NAME: str = "Numb"
COLLAB_FILTERING_ARTIST_NAME: str = "Linkin Park"
