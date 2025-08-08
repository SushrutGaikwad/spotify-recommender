import numpy as np
import dask.dataframe as dd
from scipy.sparse import csr_matrix, save_npz

from pathlib import Path
from loguru import logger

from spotify_recommender.config import (
    USER_HISTORY_DATA,
    COLLAB_TRACK_IDS_NPY,
    COLLAB_INTERACTION_MATRIX_NPZ,
)


class InteractionMatrixBuilder:
    def __init__(self, user_history_path: Path) -> None:
        """Initiates an `InteractionMatrixBuilder` object.

        Args:
            user_history_path (Path): Path of the user listening history data.
        """
        logger.info("Instantiating an `InteractionMatrixBuilder` object...")
        self.user_history_path = user_history_path
        logger.info("`InteractionMatrixBuilder` object successfully instantiated.")

    def load_history(self) -> dd.DataFrame:
        """Loads the user listening history data.

        Returns:
            dd.DataFrame: User listening history data.
        """
        try:
            logger.info("Running the method `load_history`...")
            ddf = dd.read_csv(self.user_history_path, assume_missing=True)
            logger.info("User listening history loaded successfully.")
            return ddf
        except Exception as e:
            logger.error(f"Unexpected error in `load_history`: {e}.")
            raise

    def build_and_save(self) -> csr_matrix:
        """Builds the track-user interaction matrix and saves it to disk.

        Returns:
            csr_matrix: Interaction matrix in CSR format.
        """
        try:
            logger.info("Running the method `build_and_save`...")
            ddf = self.load_history()
            ddf["playcount"] = ddf["playcount"].astype("float64")
            ddf = ddf.categorize(columns=["user_id", "track_id"])

            user_codes = ddf["user_id"].cat.codes
            track_codes = ddf["track_id"].cat.codes
            track_ids = ddf["track_id"].cat.categories.values

            logger.info(f"Saving track IDs to '{COLLAB_TRACK_IDS_NPY}'...")
            np.save(COLLAB_TRACK_IDS_NPY, track_ids, allow_pickle=True)

            ddf = ddf.assign(user_idx=user_codes, track_idx=track_codes)

            logger.info("Aggregating playcounts...")
            grouped = ddf.groupby(["track_idx", "user_idx"])["playcount"].sum().reset_index()
            grouped = grouped.compute()

            row_idx = grouped["track_idx"].to_numpy()
            col_idx = grouped["user_idx"].to_numpy()
            vals = grouped["playcount"].to_numpy(dtype="float64")

            n_tracks = int(grouped["track_idx"].max()) + 1
            n_users = int(grouped["user_idx"].max()) + 1

            logger.info(f"Creating CSR matrix of shape ({n_tracks}, {n_users})...")
            mat = csr_matrix((vals, (row_idx, col_idx)), shape=(n_tracks, n_users))

            logger.info(f"Saving interaction matrix to '{COLLAB_INTERACTION_MATRIX_NPZ}'...")
            save_npz(COLLAB_INTERACTION_MATRIX_NPZ, mat)
            logger.info("Interaction matrix saved successfully.")
            return mat
        except Exception as e:
            logger.error(f"Unexpected error in `build_and_save`: {e}.")
            raise


if __name__ == "__main__":
    interaction_matrix_builder = InteractionMatrixBuilder(user_history_path=USER_HISTORY_DATA)
    interaction_matrix = interaction_matrix_builder.build_and_save()
