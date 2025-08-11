"""Feature preparation for the hybrid recommender.

This module prepares content-based features for the *filtered* songs universe that
collaborative filtering operates on. It:
1) Loads the collab-filtered songs CSV and sorts it by `track_id` to mirror the CF index order.
2) Loads the already-trained CBF transformer.
3) Transforms the sorted songs to produce a sparse feature matrix (`spmatrix`).
4) Saves the sorted CSV and the aligned sparse feature matrix for hybrid use.
"""

import joblib
import pandas as pd

from pathlib import Path
from loguru import logger
from tqdm.auto import tqdm
from scipy.sparse import spmatrix, save_npz
from sklearn.compose import ColumnTransformer

from spotify_recommender.config import (
    COLLAB_FILTERED_SONGS_DATA,
    HYBRID_FILTERED_SONGS_SORTED,
    HYBRID_TRANSFORMED_CBF_NPZ,
    TRAINED_TRANSFORMER_CBF,
)


class FeaturePreparer:
    """Prepares CBF features aligned to the CF index universe.

    This class ensures the row order used for content-based features matches the
    collaborative filtering index (derived from `track_id`). Doing this here avoids
    alignment bugs later when combining CBF and CF similarity scores.

    Attributes:
        filtered_songs_path (Path): Path to the collab-filtered songs CSV.
        trained_transformer_path (Path): Path to the trained CBF transformer (.joblib).
    """

    def __init__(self, filtered_songs_path: Path, trained_transformer_path: Path) -> None:
        """Initiates a `FeaturePreparer` object.

        Args:
            filtered_songs_path (Path): Path of the collab-filtered songs CSV.
            trained_transformer_path (Path): Path of the trained CBF transformer (.joblib).
        """
        logger.info("Instantiating a `FeaturePreparer` object...")
        self.filtered_songs_path = filtered_songs_path
        self.trained_transformer_path = trained_transformer_path
        logger.info("`FeaturePreparer` object successfully instantiated.")

    def load_inputs(self) -> tuple[pd.DataFrame, ColumnTransformer]:
        """Loads the filtered songs and the trained transformer, then sorts by `track_id`.

        Returns:
            tuple[pd.DataFrame, ColumnTransformer]:
                - **sorted_songs_df** (`pd.DataFrame`): Collab-filtered songs sorted by `track_id`.
                  Expected to include at least the columns used by the trained transformer
                  (e.g., `name`, `artist`, `tags`, etc.) and the key `track_id`.
                - **trained_transformer** (`ColumnTransformer`): Fitted CBF transformer loaded
                  from disk; typically includes encoders and scalers (OHE, TF-IDF, etc.).

        Raises:
            FileNotFoundError: If the songs CSV or transformer file is not found.
            KeyError: If `track_id` is missing from the filtered songs.
            Exception: For any unexpected errors during load.
        """
        try:
            logger.info("Running the method `load_inputs`...")
            songs_df = pd.read_csv(self.filtered_songs_path)
            if "track_id" not in songs_df.columns:
                raise KeyError("'track_id' column missing in filtered songs.")
            # Sort by track_id so the row order mirrors CF row order (lexicographic by `track_id`)
            songs_df = songs_df.sort_values("track_id").reset_index(drop=True)
            transformer: ColumnTransformer = joblib.load(self.trained_transformer_path)
            logger.info(
                f"Loaded & sorted songs_df {songs_df.shape}; loaded transformer '{self.trained_transformer_path}'."
            )
            return songs_df, transformer
        except FileNotFoundError as e:
            logger.error(f"File not found in `load_inputs`: {e}.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in `load_inputs`: {e}.")
            raise

    def transform(self, songs_df: pd.DataFrame, transformer: ColumnTransformer) -> spmatrix:
        """Transforms the sorted filtered songs using the trained CBF transformer.

        Args:
            songs_df (pd.DataFrame): Collab-filtered songs **sorted by `track_id`**.
            transformer (ColumnTransformer): Trained CBF transformer that accepts the columns
                present in `songs_df` and outputs a (typically) sparse feature matrix.

        Returns:
            spmatrix: A SciPy sparse matrix (CSR/CSC; base type `spmatrix`) of shape
                `(n_songs, n_features_cbf)` representing content-based features aligned to
                the sorted `songs_df` row order.

        Raises:
            Exception: For unexpected errors during transformation.
        """
        try:
            logger.info("Running the method `transform`...")
            with tqdm(total=1, desc="Transforming hybrid CBF features", unit="step") as pbar:
                X = transformer.transform(songs_df)
                pbar.update(1)
            logger.info("Hybrid CBF features generated successfully.")
            return X
        except Exception as e:
            logger.error(f"Unexpected error in `transform`: {e}.")
            raise

    def run(self) -> spmatrix:
        """Orchestrates sorting, transforming, and saving hybrid CBF features.

        Workflow:
            1) `load_inputs()` to get the sorted songs and trained transformer.
            2) `transform()` to produce the aligned sparse feature matrix.
            3) Save the sorted CSV to `HYBRID_FILTERED_SONGS_SORTED`.
            4) Save the aligned feature matrix to `HYBRID_TRANSFORMED_CBF_NPZ`.

        Returns:
            spmatrix: The aligned sparse feature matrix produced by `transform()`.

        Raises:
            Exception: For unexpected errors across the pipeline.
        """
        try:
            logger.info("Running the method `run`...")
            songs_df, transformer = self.load_inputs()
            X = self.transform(songs_df, transformer)

            logger.info(f"Saving sorted songs to '{HYBRID_FILTERED_SONGS_SORTED}'...")
            songs_df.to_csv(HYBRID_FILTERED_SONGS_SORTED, index=False)

            logger.info(f"Saving hybrid CBF features to '{HYBRID_TRANSFORMED_CBF_NPZ}'...")
            save_npz(HYBRID_TRANSFORMED_CBF_NPZ, X)
            logger.info("Hybrid CBF features saved successfully.")
            return X
        except Exception as e:
            logger.error(f"Unexpected error in `run`: {e}.")
            raise


if __name__ == "__main__":
    prep = FeaturePreparer(
        filtered_songs_path=COLLAB_FILTERED_SONGS_DATA,
        trained_transformer_path=TRAINED_TRANSFORMER_CBF,
    )
    _ = prep.run()
