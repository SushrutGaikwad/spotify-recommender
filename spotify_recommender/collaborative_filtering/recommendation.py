import numpy as np
import pandas as pd

from pathlib import Path
from loguru import logger
from tqdm.auto import tqdm
from scipy.sparse import load_npz, spmatrix
from sklearn.metrics.pairwise import cosine_similarity

from spotify_recommender.config import (
    COLLAB_FILTERED_SONGS_DATA,
    COLLAB_TRACK_IDS_NPY,
    COLLAB_INTERACTION_MATRIX_NPZ,
    COLLAB_K,
    COLLAB_FILTERING_SONG_NAME,
    COLLAB_FILTERING_ARTIST_NAME,
)


class Recommender:
    def __init__(
        self,
        filtered_songs_path: Path,
        track_ids_path: Path,
        interaction_matrix_path: Path,
    ) -> None:
        """Initiates a `Recommender` object.

        Args:
            filtered_songs_path (Path): Path to the filtered songs data.
            track_ids_path (Path): Path to the saved track IDs file.
            interaction_matrix_path (Path): Path to the saved interaction matrix file.
        """
        logger.info("Instantiating a `Recommender` object...")
        self.filtered_songs_path = filtered_songs_path
        self.track_ids_path = track_ids_path
        self.interaction_matrix_path = interaction_matrix_path
        self.k: int | None = None
        logger.info("`Recommender` object successfully instantiated.")

    def load_artifacts(self) -> tuple[pd.DataFrame, np.ndarray, spmatrix]:
        """Loads the filtered songs data, track IDs, and interaction matrix.

        Returns:
            tuple[pd.DataFrame, np.ndarray, spmatrix]: Filtered songs data, track IDs, and interaction matrix.
        """
        try:
            logger.info("Running the method `load_artifacts`...")
            songs = pd.read_csv(self.filtered_songs_path)
            track_ids = np.load(self.track_ids_path, allow_pickle=True)
            mat = load_npz(self.interaction_matrix_path)
            logger.info("Artifacts loaded successfully.")
            return songs, track_ids, mat
        except Exception as e:
            logger.error(f"Unexpected error in `load_artifacts`: {e}.")
            raise

    def recommend(self, song_name: str, artist_name: str, k: int = COLLAB_K) -> pd.DataFrame:
        """The orchestrator method that generates collaborative filtering recommendations.

        Args:
            song_name (str): Name of the song to base recommendations on.
            artist_name (str): Artist of the song.
            k (int, optional): Number of recommendations to return. Defaults to `COLLAB_K`.

        Returns:
            pd.DataFrame: Top-`k` recommended songs.
        """
        try:
            logger.info("Running the method `recommend`...")
            self.k = k
            songs, track_ids, mat = self.load_artifacts()

            song_name = song_name.lower()
            artist_name = artist_name.lower()

            song_row = songs.loc[(songs["name"] == song_name) & (songs["artist"] == artist_name)]
            if song_row.empty:
                raise ValueError(f"Song '{song_name}' by '{artist_name}' not found in dataset.")

            input_track_id = song_row["track_id"].values.item()
            try:
                ind = int(np.where(track_ids == input_track_id)[0].item())
            except Exception:
                raise ValueError(f"track_id '{input_track_id}' missing from CF index.")

            input_vec = mat[ind]

            with tqdm(total=1, desc="Computing similarities", unit="step") as pbar:
                sims = cosine_similarity(input_vec, mat).ravel()
                pbar.update(1)

            order = np.argsort(sims)[::-1]
            order = order[order != ind]
            top_idx = order[:k]

            rec_track_ids = track_ids[top_idx]
            rec_scores = sims[top_idx]
            scores_df = pd.DataFrame({"track_id": rec_track_ids, "score": rec_scores})

            top_k = (
                songs.loc[songs["track_id"].isin(rec_track_ids)]
                .merge(scores_df, on="track_id")
                .sort_values("score", ascending=False)
                .drop(columns=["track_id", "score"])
                .reset_index(drop=True)
            )
            logger.info(f"Top-{k} recommendations generated successfully.")
            return top_k
        except Exception as e:
            logger.error(f"Unexpected error in `recommend`: {e}.")
            raise


if __name__ == "__main__":
    try:
        logger.info("Starting collaborative filtering recommendation process...")
        recommender = Recommender(
            filtered_songs_path=COLLAB_FILTERED_SONGS_DATA,
            track_ids_path=COLLAB_TRACK_IDS_NPY,
            interaction_matrix_path=COLLAB_INTERACTION_MATRIX_NPZ,
        )
        df = recommender.recommend(
            song_name=COLLAB_FILTERING_SONG_NAME, artist_name=COLLAB_FILTERING_ARTIST_NAME
        )
        logger.info(f"Top-{recommender.k} collaborative recommendations:\n{df}")
        logger.info("Collaborative filtering recommendation process completed successfully!")
    except Exception as e:
        logger.critical(f"Collaborative filtering recommendation process failed: {e}.")
