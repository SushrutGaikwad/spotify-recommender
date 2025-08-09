import numpy as np
import pandas as pd

from pathlib import Path
from loguru import logger
from tqdm.auto import tqdm
from scipy.sparse import load_npz, spmatrix
from sklearn.metrics.pairwise import cosine_similarity

from spotify_recommender.config import (
    CLEANED_SONGS_DATA,
    TRANSFORMED_SONGS_DATA_CBF,
    CONTENT_BASED_FILTERING_RECO_SONG_NAME,
    CONTENT_BASED_FILTERING_RECO_K,
)


class Recommender:
    def __init__(self, cleaned_data_path: Path, transformed_data_path: Path) -> None:
        """Initiates a `Recommender` object.

        Args:
            cleaned_data_path (Path): Path of the cleaned data.
            transformed_data_path (Path): Path of the transformed data.
        """
        logger.info("Instantiating a `Recommender` object...")
        self.cleaned_data_path = cleaned_data_path
        self.transformed_data_path = transformed_data_path
        self.k = None
        logger.info("`Recommender` object successfully instantiated.")

    def load_cleaned_data(self) -> pd.DataFrame:
        """Loads the cleaned data.

        Returns:
            pd.DataFrame: Cleaned data.
        """
        try:
            logger.info("Running the method `load_cleaned_data`...")
            logger.info(f"Loading cleaned data from '{self.cleaned_data_path}'...")
            cleaned_data = pd.read_csv(self.cleaned_data_path)
            logger.info(
                f"Cleaned data loaded successfully with {cleaned_data.shape[0]} rows and {cleaned_data.shape[1]} columns."
            )
            return cleaned_data
        except FileNotFoundError:
            logger.error(f"File '{self.cleaned_data_path}' not found.")
            raise
        except pd.errors.EmptyDataError:
            logger.error("CSV file is empty.")
            raise
        except pd.errors.ParserError:
            logger.error("Error parsing the CSV file.")
            raise

    def load_transformed_data(self) -> spmatrix:
        """Loads the transformed data.

        Returns:
            spmatrix: Transformed data.
        """
        try:
            logger.info("Running the method `load_transformed_data`...")
            logger.info(f"Loading transformed data from '{self.transformed_data_path}'...")
            data = load_npz(self.transformed_data_path)
            logger.info("Transformed data loaded successfully.")
            return data
        except FileNotFoundError:
            logger.error(f"File '{self.transformed_data_path}' not found.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in `load_transformed_data`: {e}.")
            raise

    def calculate_similarity_scores(self, input_song: spmatrix, all_songs: spmatrix) -> np.ndarray:
        """Calculates similarity scores of a song with all the songs using cosine similarity.

        Args:
            input_song (spmatrix): The input song.
            all_songs (spmatrix): All the songs.

        Returns:
            np.ndarray: Similarity scores.
        """
        try:
            logger.info("Running the method `calculate_similarity_scores`...")
            with tqdm(total=1, desc="Calculating Similarity", unit="step") as pbar:
                similarity_scores = cosine_similarity(input_song, all_songs)
                pbar.update(1)
            logger.info("Similarity scores calculated successfully.")
            return similarity_scores
        except Exception as e:
            logger.error(f"Error in `calculate_similarity_scores`: {e}.")
            raise

    def recommend(self, song_name: str, k: int) -> pd.DataFrame:
        """The orchestrator method that recommends songs using content-based filtering.

        Args:
            song_name (str): Name of the song to base the recommendation on.
            k (int): Number of songs to recommend.

        Returns:
            pd.DataFrame: The top `k` song recommendations.
        """
        try:
            logger.info("Running the method `recommend`...")
            self.k = k
            cleaned_data = self.load_cleaned_data()
            transformed_data = self.load_transformed_data()

            song_name = song_name.lower()
            song_row = cleaned_data.loc[cleaned_data["name"] == song_name]

            if song_row.empty:
                logger.warning(f"Song '{song_name}' not found in dataset.")
                raise ValueError(f"Song '{song_name}' not found in dataset.")

            song_idx = song_row.index[0]
            input_song_vector = transformed_data[song_idx].reshape(1, -1)

            similarity_scores = self.calculate_similarity_scores(
                input_song=input_song_vector, all_songs=transformed_data
            )

            # top_k_most_similar_songs_idxs = np.argsort(similarity_scores.ravel())[::-1][: k + 1]
            ordered = np.argsort(similarity_scores.ravel())[::-1]
            top_k_most_similar_songs_idxs = ordered[ordered != song_idx][:k]
            top_k_most_similar_songs_name = cleaned_data.loc[top_k_most_similar_songs_idxs]
            top_k_most_similar_songs_df = top_k_most_similar_songs_name[
                ["name", "artist", "spotify_preview_url"]
            ].reset_index(drop=True)
            logger.info(f"Top-{k} recommendations generated successfully.")
            return top_k_most_similar_songs_df
        except ValueError as e:
            logger.error(f"ValueError in `recommend`: {e}.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in `recommend`: {e}.")
            raise


if __name__ == "__main__":
    try:
        logger.info("Starting content-based filtering recommendation process...")
        recommender = Recommender(
            cleaned_data_path=CLEANED_SONGS_DATA, transformed_data_path=TRANSFORMED_SONGS_DATA_CBF
        )
        recommendations = recommender.recommend(
            song_name=CONTENT_BASED_FILTERING_RECO_SONG_NAME, k=CONTENT_BASED_FILTERING_RECO_K
        )
        logger.info(f"Top-{recommender.k} song recommendations:")
        logger.info(f"\n{recommendations}")
        logger.info("Content-based filtering recommendation process completed successfully!")
    except Exception as e:
        logger.critical(f"Content-based filtering recommendation process failed: {e}.")
