"""Hybrid recommender (CBF + CF) with weighted similarity fusion.

This module:
1) Loads the CF universe (track_ids + interaction matrix) and the CBF features aligned to the
   collab-filtered songs (sorted by `track_id`).
2) Computes cosine similarities for both CBF and CF for the seed item.
3) Normalizes each similarity vector (min-max) to mitigate scale differences.
4) Reindexes CF similarities to the songs DataFrame order, ensuring 1:1 alignment.
5) Combines (element-wise) using weights w_cb and w_cf, excludes the seed, and returns top-k.
"""

import numpy as np
import pandas as pd

from pathlib import Path
from loguru import logger
from tqdm.auto import tqdm
from scipy.sparse import load_npz, spmatrix
from sklearn.metrics.pairwise import cosine_similarity

from spotify_recommender.config import (
    # Paths
    HYBRID_FILTERED_SONGS_SORTED,
    HYBRID_TRANSFORMED_CBF_NPZ,
    COLLAB_TRACK_IDS_NPY,
    COLLAB_INTERACTION_MATRIX_NPZ,
    # Params
    HYBRID_K,
    HYBRID_SONG_NAME,
    HYBRID_ARTIST_NAME,
    HYBRID_WEIGHT_CBF,
    HYBRID_WEIGHT_CF,
)


class Recommender:
    """Hybrid recommender that fuses CBF and CF similarities via a weighted sum.

    Attributes:
        filtered_songs_path (Path): Path to the **sorted** collab-filtered songs CSV.
        hybrid_cbf_features_path (Path): Path to the aligned CBF features (npz; `spmatrix`).
        track_ids_path (Path): Path to the CF `track_ids.npy` aligned to CF matrix rows.
        interaction_matrix_path (Path): Path to the CF interaction matrix (npz; `spmatrix`).

        k (int | None): Number of recommendations used in the last `recommend()` call.
        weight_cbf (float | None): CBF weight used in the last `recommend()` call.
        weight_cf (float | None): CF weight used in the last `recommend()` call.
    """

    def __init__(
        self,
        filtered_songs_path: Path,
        hybrid_cbf_features_path: Path,
        track_ids_path: Path,
        interaction_matrix_path: Path,
    ) -> None:
        """Initiates a `Recommender` object.

        Args:
            filtered_songs_path (Path): Path of the **sorted** collab-filtered songs CSV.
            hybrid_cbf_features_path (Path): Path of the aligned CBF features (npz).
            track_ids_path (Path): Path of the saved CF track IDs (npy).
            interaction_matrix_path (Path): Path of the CF interaction matrix (npz).
        """
        logger.info("Instantiating a `Recommender` object...")
        self.filtered_songs_path = filtered_songs_path
        self.hybrid_cbf_features_path = hybrid_cbf_features_path
        self.track_ids_path = track_ids_path
        self.interaction_matrix_path = interaction_matrix_path

        self.k: int | None = None
        self.weight_cbf: float | None = None
        self.weight_cf: float | None = None
        logger.info("`Recommender` object successfully instantiated.")

    def load_artifacts(self) -> tuple[pd.DataFrame, spmatrix, np.ndarray, spmatrix]:
        """Loads all artifacts required for hybrid recommendation.

        Returns:
            tuple[pd.DataFrame, spmatrix, np.ndarray, spmatrix]:
                - **songs_df** (`pd.DataFrame`): Sorted collab-filtered songs. Must contain
                  `track_id`, `name`, and `artist`, all lowercased to match lookup style.
                - **X_cbf** (`spmatrix`): Sparse CBF feature matrix aligned to `songs_df` row order.
                - **track_ids** (`np.ndarray`): CF track IDs aligned to rows of `X_cf`.
                - **X_cf** (`spmatrix`): Sparse CF interaction matrix (tracks × users).

        Raises:
            ValueError: If row counts/shapes don't align (e.g., CBF rows ≠ songs rows).
            KeyError: If `track_id` is missing.
            Exception: For unexpected load errors.
        """
        try:
            logger.info("Running the method `load_artifacts`...")
            songs_df = pd.read_csv(self.filtered_songs_path)
            X_cbf = load_npz(self.hybrid_cbf_features_path)
            track_ids = np.load(self.track_ids_path, allow_pickle=True)
            X_cf = load_npz(self.interaction_matrix_path)

            logger.info(
                "Artifacts loaded: "
                f"songs_df={songs_df.shape}, X_cbf={X_cbf.shape}, "
                f"track_ids={len(track_ids)}, X_cf={X_cf.shape}"
            )

            # Sanity checks local to each source
            if X_cbf.shape[0] != songs_df.shape[0]:
                raise ValueError(
                    "Row mismatch: hybrid CBF features and filtered songs must align. "
                    f"Got X_cbf={X_cbf.shape[0]} vs songs_df={songs_df.shape[0]}."
                )
            if X_cf.shape[0] != len(track_ids):
                raise ValueError(
                    "Row mismatch: CF interaction matrix rows must align with track_ids length. "
                    f"Got X_cf={X_cf.shape[0]} vs track_ids={len(track_ids)}."
                )
            if "track_id" not in songs_df.columns:
                raise KeyError("'track_id' column missing in filtered songs.")
            if not songs_df["track_id"].is_monotonic_increasing:
                raise ValueError(
                    "Expected songs_df to be sorted by track_id for hybrid alignment."
                )

            return songs_df, X_cbf, track_ids, X_cf
        except Exception as e:
            logger.error(f"Unexpected error in `load_artifacts`: {e}.")
            raise

    def _find_input_indices(
        self, song_name: str, artist_name: str, songs_df: pd.DataFrame, track_ids: np.ndarray
    ) -> tuple[int, int, str]:
        """Finds the seed indices in both CBF (songs_df order) and CF (track_ids order).

        Args:
            song_name (str): Seed song name (case-insensitive; will be lowercased).
            artist_name (str): Seed artist name (case-insensitive; will be lowercased).
            songs_df (pd.DataFrame): Sorted collab-filtered songs DataFrame.
            track_ids (np.ndarray): CF track IDs aligned to CF rows.

        Returns:
            tuple[int, int, str]:
                - **songs_idx** (`int`): Row index of the seed in `songs_df` / `X_cbf`.
                - **cf_idx** (`int`): Row index of the seed in `X_cf` (via `track_ids`).
                - **input_track_id** (`str` | `int`): The seed `track_id` value.

        Raises:
            ValueError: If the (name, artist) pair isn't found in `songs_df`, or if the
                corresponding `track_id` is missing from the CF index.
        """
        song_row = songs_df.loc[
            (songs_df["name"] == song_name.lower()) & (songs_df["artist"] == artist_name.lower())
        ]
        if song_row.empty:
            raise ValueError(f"Song '{song_name}' by '{artist_name}' not found in filtered songs.")

        songs_idx = song_row.index[0]
        input_track_id = song_row["track_id"].values.item()

        try:
            cf_idx = int(np.where(track_ids == input_track_id)[0].item())
        except Exception:
            raise ValueError(f"track_id '{input_track_id}' missing from CF index.")

        return songs_idx, cf_idx, input_track_id

    @staticmethod
    def _cosine_row_to_matrix(row_vec: spmatrix, mat: spmatrix, desc: str) -> np.ndarray:
        """Computes cosine similarity between a single row vector and all rows in a matrix.

        Args:
            row_vec (spmatrix): Sparse row vector of shape `(1, n_features)` or `(1, n_users)`.
            mat (spmatrix): Sparse matrix to compare against (CBF features or CF interaction).
            desc (str): Description used for tqdm progress bar.

        Returns:
            np.ndarray: 1D array of cosine similarities of shape `(n_rows_in_mat,)`.
        """
        with tqdm(total=1, desc=desc, unit="step") as pbar:
            sims = cosine_similarity(row_vec, mat).ravel()
            pbar.update(1)
        return sims

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Min-max normalizes a similarity vector, safely handling constant arrays.

        Args:
            scores (np.ndarray): Raw similarity scores.

        Returns:
            np.ndarray: Normalized scores in `[0, 1]`. If all values are equal, returns zeros.
        """
        smin = float(np.min(scores))
        smax = float(np.max(scores))
        if smax <= smin:
            return np.zeros_like(scores)
        return (scores - smin) / (smax - smin)

    @staticmethod
    def _weighted_sum(cbf_scores: np.ndarray, cf_scores: np.ndarray, w_cbf: float, w_cf: float):
        """Returns the weighted sum of two aligned similarity vectors.

        Args:
            cbf_scores (np.ndarray): Normalized CBF similarities (aligned order/shape).
            cf_scores (np.ndarray): Normalized CF similarities (aligned order/shape).
            w_cbf (float): Weight for the CBF component.
            w_cf (float): Weight for the CF component.

        Returns:
            np.ndarray: Weighted similarity scores (same shape as inputs).
        """
        return (w_cbf * cbf_scores) + (w_cf * cf_scores)

    def recommend(
        self,
        song_name: str,
        artist_name: str,
        k: int = HYBRID_K,
        weight_cbf: float = HYBRID_WEIGHT_CBF,
        weight_cf: float = HYBRID_WEIGHT_CF,
    ) -> pd.DataFrame:
        """Generates top-`k` hybrid recommendations (CBF ⊕ CF).

        Args:
            song_name (str): Seed song name (case-insensitive; matched on lowercase).
            artist_name (str): Seed artist name (case-insensitive; matched on lowercase).
            k (int, optional): Number of recommendations to return. Defaults to `HYBRID_K`.
            weight_cbf (float, optional): Weight for CBF similarities. Defaults to `HYBRID_WEIGHT_CBF`.
            weight_cf (float, optional): Weight for CF similarities. Defaults to `HYBRID_WEIGHT_CF`.

        Returns:
            pd.DataFrame: Top-`k` recommendations with columns:
                - `name` (`str`): Song name (lowercased in your pipeline).
                - `artist` (`str`): Artist name (lowercased).
                - `spotify_preview_url` (`str` | `NaN`): Preview URL if available.

        Raises:
            ValueError: If artifacts are misaligned, if the seed is not found,
                or if shapes still mismatch after alignment.
            Exception: For any unexpected runtime errors.
        """
        try:
            logger.info("Running the method `recommend`...")
            self.k, self.weight_cbf, self.weight_cf = k, weight_cbf, weight_cf

            songs_df, X_cbf, track_ids_full, X_cf_full = self.load_artifacts()
            songs_idx, cf_idx_full, input_tid = self._find_input_indices(
                song_name, artist_name, songs_df, track_ids_full
            )

            # ---- 1) Compute similarities on each universe ----
            cbf_row = X_cbf[songs_idx].reshape(1, -1)  # songs_df row order
            cbf_sims = self._cosine_row_to_matrix(cbf_row, X_cbf, desc="CBF similarities")
            cbf_sims_norm = self._normalize_scores(cbf_sims)

            cf_row_full = X_cf_full[cf_idx_full]
            cf_sims_full = self._cosine_row_to_matrix(
                cf_row_full, X_cf_full, desc="CF similarities"
            )
            cf_sims_full_norm = self._normalize_scores(cf_sims_full)

            # ---- 2) Align CF scores to songs_df order (subset + reorder) ----
            songs_tids = songs_df["track_id"].to_numpy()
            pos_by_tid = {tid: idx for idx, tid in enumerate(track_ids_full)}
            try:
                cf_positions_for_songs = np.array(
                    [pos_by_tid[tid] for tid in songs_tids], dtype=int
                )
            except KeyError as e:
                raise ValueError(
                    f"A track_id in songs_df is missing from CF index: {e}. "
                    "Rebuild artifacts to ensure alignment."
                )
            cf_sims_norm = cf_sims_full_norm[cf_positions_for_songs]

            # Seed index in aligned (songs_df) space
            try:
                seed_aligned_idx = int(np.where(songs_tids == input_tid)[0].item())
            except Exception:
                raise ValueError("Failed to locate seed track in aligned index space.")

            # ---- 3) Combine (now same shape/order) ----
            if cbf_sims_norm.shape != cf_sims_norm.shape:
                raise ValueError(
                    f"Shape mismatch after alignment: CBF {cbf_sims_norm.shape} vs CF {cf_sims_norm.shape}."
                )
            weighted = self._weighted_sum(cbf_sims_norm, cf_sims_norm, weight_cbf, weight_cf)

            # ---- 4) Rank (exclude seed) ----
            order = np.argsort(weighted)[::-1]
            order = order[order != seed_aligned_idx]
            top_idx = order[:k]

            rec_track_ids = songs_tids[top_idx]
            rec_scores = weighted[top_idx]
            scores_df = pd.DataFrame({"track_id": rec_track_ids, "score": rec_scores})

            top_k = (
                songs_df.loc[songs_df["track_id"].isin(rec_track_ids)]
                .merge(scores_df, on="track_id")
                .sort_values("score", ascending=False)
                .drop(columns=["track_id", "score"])
                .reset_index(drop=True)
            )
            logger.info(f"Top-{k} hybrid recommendations generated successfully.")
            return top_k
        except Exception as e:
            logger.error(f"Unexpected error in `recommend`: {e}.")
            raise


if __name__ == "__main__":
    try:
        logger.info("Starting hybrid recommendation process...")
        recommender = Recommender(
            filtered_songs_path=HYBRID_FILTERED_SONGS_SORTED,
            hybrid_cbf_features_path=HYBRID_TRANSFORMED_CBF_NPZ,
            track_ids_path=COLLAB_TRACK_IDS_NPY,
            interaction_matrix_path=COLLAB_INTERACTION_MATRIX_NPZ,
        )
        df = recommender.recommend(
            song_name=HYBRID_SONG_NAME,
            artist_name=HYBRID_ARTIST_NAME,
            k=HYBRID_K,
            weight_cbf=HYBRID_WEIGHT_CBF,
            weight_cf=HYBRID_WEIGHT_CF,
        )
        logger.info(f"Top-{recommender.k} hybrid recommendations:\n{df}")
        logger.info("Hybrid recommendation process completed successfully!")
    except Exception as e:
        logger.critical(f"Hybrid recommendation process failed: {e}.")
