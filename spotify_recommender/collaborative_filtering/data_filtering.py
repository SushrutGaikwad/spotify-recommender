import numpy as np
import pandas as pd
import dask.dataframe as dd

from pathlib import Path
from loguru import logger
from tqdm.auto import tqdm

from spotify_recommender.config import (
    CLEANED_SONGS_DATA,
    USER_HISTORY_DATA,
    COLLAB_FILTERED_SONGS_DATA,
)


class SongsFilterer:
    def __init__(self, cleaned_songs_path: Path, user_history_path: Path) -> None:
        """Initiates a `SongsFilterer` object.

        Args:
            cleaned_songs_path (Path): Path of the cleaned songs data.
            user_history_path (Path): Path of the user listening history data.
        """
        logger.info("Instantiating a `SongsFilterer` object...")
        self.cleaned_songs_path = cleaned_songs_path
        self.user_history_path = user_history_path
        logger.info("`SongsFilterer` object successfully instantiated.")

    def load_sources(self) -> tuple[pd.DataFrame, dd.DataFrame]:
        """Loads the cleaned songs data and user listening history data.

        Returns:
            tuple[pd.DataFrame, dd.DataFrame]: Cleaned songs data (pandas) and user history data (dask).
        """
        try:
            logger.info("Running the method `load_sources`...")
            songs_df = pd.read_csv(self.cleaned_songs_path)
            history_ddf = dd.read_csv(self.user_history_path, assume_missing=True)
            logger.info("Sources loaded successfully.")
            return songs_df, history_ddf
        except Exception as e:
            logger.error(f"Unexpected error in `load_sources`: {e}.")
            raise

    def get_unique_track_ids(self, history_ddf: dd.DataFrame) -> np.ndarray:
        """Gets the unique track IDs from the user listening history.

        Args:
            history_ddf (dd.DataFrame): User listening history data.

        Returns:
            np.ndarray: Unique track IDs.
        """
        try:
            logger.info("Running the method `get_unique_track_ids`...")
            track_ids = history_ddf["track_id"].dropna().unique().compute()
            logger.info("Unique track IDs retrieved successfully.")
            return track_ids.values if hasattr(track_ids, "values") else track_ids
        except Exception as e:
            logger.error(f"Unexpected error in `get_unique_track_ids`: {e}.")
            raise

    def filter_songs(self, songs_df: pd.DataFrame, track_ids: np.ndarray) -> pd.DataFrame:
        """Filters the songs data to only include tracks present in the listening history.

        Args:
            songs_df (pd.DataFrame): Cleaned songs data.
            track_ids (np.ndarray): Unique track IDs from listening history.

        Returns:
            pd.DataFrame: Filtered songs data.
        """
        try:
            logger.info("Running the method `filter_songs`...")
            with tqdm(total=1, desc="Filtering songs", unit="step") as pbar:
                filtered = songs_df[songs_df["track_id"].isin(track_ids)].reset_index(drop=True)
                pbar.update(1)
            logger.info("Songs filtered successfully.")
            return filtered
        except KeyError as e:
            logger.error(f"KeyError in `filter_songs`: {e}.")
            raise

    def run(self) -> pd.DataFrame:
        """The orchestrator method that filters the songs dataset.

        Returns:
            pd.DataFrame: Filtered songs data.
        """
        try:
            logger.info("Running the method `run`...")
            songs_df, history_ddf = self.load_sources()
            track_ids = self.get_unique_track_ids(history_ddf)
            filtered = self.filter_songs(songs_df, track_ids)
            logger.info(f"Saving filtered songs to '{COLLAB_FILTERED_SONGS_DATA}'...")
            filtered.to_csv(COLLAB_FILTERED_SONGS_DATA, index=False)
            logger.info("Filtered songs saved successfully.")
            return filtered
        except Exception as e:
            logger.error(f"Unexpected error in `run`: {e}.")
            raise


if __name__ == "__main__":
    filterer = SongsFilterer(
        cleaned_songs_path=CLEANED_SONGS_DATA,
        user_history_path=USER_HISTORY_DATA,
    )
    filtered_songs = filterer.run()
