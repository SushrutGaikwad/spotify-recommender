import pandas as pd
import streamlit as st

from pathlib import Path
from loguru import logger
from tqdm.auto import tqdm
from scipy.sparse import load_npz, spmatrix

from spotify_recommender.content_based_filtering.recommendation import (
    Recommender,
)
from spotify_recommender.config import CLEANED_SONGS_DATA, TRANSFORMED_SONGS_DATA_CBF


class SpotifyRecommenderApp:
    def __init__(self, cleaned_data_path: Path, transformed_data_path: Path) -> None:
        logger.info("Instantiating a `SpotifyRecommenderApp` object...")
        self.cleaned_data_path = cleaned_data_path
        self.transformed_data_path = transformed_data_path
        self.recommender = Recommender(
            cleaned_data_path=self.cleaned_data_path,
            transformed_data_path=self.transformed_data_path,
        )
        logger.info("`SpotifyRecommenderApp` object successfully instantiated.")

    def load_cleaned_data(self) -> pd.DataFrame:
        """Loads the cleaned data.

        Returns:
            pd.DataFrame: Cleaned data.
        """
        try:
            logger.info("Running the method `load_cleaned_data`...")
            logger.info(f"Loading cleaned data from '{self.cleaned_data_path}'...")
            cleaned_data = pd.read_csv(self.cleaned_data_path)
            return cleaned_data
        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError: {e}.")
            st.error(f"Error: Missing required files! {e}.")
        except Exception as e:
            logger.error(f"Unexpected error in `load_data`: {e}.")
            st.error("An error occurred while loading the data.")

    def load_transformed_data(self) -> spmatrix:
        """Loads the transformed data.

        Returns:
            spmatrix: Transformed data.
        """
        try:
            logger.info("Running the method `load_transformed_data`...")
            logger.info(f"Loading transformed data from '{self.transformed_data_path}'...")
            transformed_data = load_npz(self.transformed_data_path)
            return transformed_data
        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError: {e}.")
            st.error(f"Error: Missing required files! {e}.")
        except Exception as e:
            logger.error(f"Unexpected error in `load_data`: {e}.")
            st.error("An error occurred while loading the data.")

    def get_recommendations(
        self, song_name: str, k: int, cleaned_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Generates recommendations based on a song name.

        Args:
            song_name (str): Name of the song to generate recommendations from.
            k (int): The number of recommendations to generate.
            cleaned_data (pd.DataFrame): The cleaned data.

        Returns:
            pd.DataFrame: Recommendations.
        """
        try:
            logger.info("Running the method `get_recommendations`...")
            logger.info(f'Fetching k={k} recommendations for the song "{song_name}"...')
            song_name_lower = song_name.lower()

            if (cleaned_data["name"] == song_name_lower).any():
                with tqdm(total=1, desc="Generating Recommendations", unit="step") as pbar:
                    recommendations = self.recommender.recommend(song_name=song_name, k=k)
                    pbar.update(1)

                logger.info(f"Successfully generated {k} recommendations.")
                return recommendations
            else:
                logger.warning(f'The song "{song_name}" is not found in the dataset.')
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error in `get_recommendations`: {e}.")
            return pd.DataFrame()

    def display_recommendations(self, recommendations: pd.DataFrame) -> None:
        """Displays recommendations.

        Args:
            recommendations (pd.DataFrame): Recommendations.
        """
        try:
            logger.info("Running the method `display_recommendations`...")

            with tqdm(
                total=len(recommendations), desc="Displaying Recommendations", unit="song"
            ) as pbar:
                for idx, recommendation in recommendations.iterrows():
                    song_name = recommendation["name"].title()
                    artist_name = recommendation["artist"].title()

                    if idx == 0:
                        st.markdown("### Currently Playing:")
                        st.markdown(f'##### "{song_name}" by "{artist_name}".')
                        st.audio(recommendation["spotify_preview_url"])
                        st.write("---")
                    elif idx == 1:
                        st.markdown("#### Next Up 🎵")
                        st.markdown(f'##### {idx}. "{song_name}" by "{artist_name}".')
                        st.audio(recommendation["spotify_preview_url"])
                        st.write("---")
                    else:
                        st.markdown(f'##### {idx}. "{song_name}" by "{artist_name}".')
                        st.audio(recommendation["spotify_preview_url"])
                        st.write("---")

                    pbar.update(1)

            logger.info("Recommendations displayed successfully.")
        except Exception as e:
            logger.error(f"Unexpected error in `display_recommendations`: {e}.")
            st.error("An error occurred while displaying recommendations.")

    def handle_recommendations(self, song_name: str, k: int, cleaned_data: pd.DataFrame) -> None:
        """Handles recommendations and displays them if they are not empty. Raises an error if the song is not found in the data.

        Args:
            song_name (str): Name of the song.
            k (int): Number of recommendations to display.
            cleaned_data (pd.DataFrame): Cleaned data.
        """
        try:
            logger.info("Running the method `handle_recommendations`...")
            if not song_name.strip():
                st.warning("Please enter a song name!")
                return

            recommendations = self.get_recommendations(
                song_name=song_name, k=k, cleaned_data=cleaned_data
            )

            if recommendations.empty:
                st.error(
                    f"""Sorry, we couldn't find the song "{song_name}" in our database. Please try another song."""
                )
            else:
                st.markdown("## We suggest you would love the following:")
                self.display_recommendations(recommendations)
            logger.info("Recommendations handled successfully.")
        except Exception as e:
            logger.error(f"Unexpected error in `handle_recommendation`: {e}.")
            st.error("Something went wrong while generating recommendations. Please try again.")

    def run(self) -> None:
        """The orchestrator method that runs the app."""
        logger.info("Running the method `run`...")
        logger.info("Running the app...")

        st.title("🎵 Welcome to the Spotify Song Recommender!")
        st.write("Start typing a song name to get suggestions. We'll recommend similar songs 😉🎧")

        # Load cleaned data
        cleaned_data = self.load_cleaned_data()

        if "name" not in cleaned_data.columns or "artist" not in cleaned_data.columns:
            st.error("The dataset must contain 'name' and 'artist' columns.")
            return

        # Drop NaNs and create list of (name, artist) pairs
        cleaned_data = cleaned_data.dropna(subset=["name", "artist"])
        cleaned_data["name"] = cleaned_data["name"].astype(str)
        cleaned_data["artist"] = cleaned_data["artist"].astype(str)

        # User types a search string
        typed_input = st.text_input("🔎 Type a song name and press enter/return:")

        # Filter entries where input matches the song name
        filtered_df = (
            cleaned_data[cleaned_data["name"].str.contains(typed_input.lower(), na=False)]
            if typed_input
            else pd.DataFrame()
        )

        # Build display labels and mapping to lowercase song names
        display_labels = []
        label_to_name_map = {}

        for _, row in filtered_df.iterrows():
            title_case_label = f'{row["name"].title()} — {row["artist"].title()}'
            display_labels.append(title_case_label)
            label_to_name_map[title_case_label] = row["name"]  # keep original lowercase song name

        selected_label = None
        if display_labels:
            selected_label = st.selectbox("🎶 Matching songs, pick yours:", sorted(display_labels))
        elif typed_input:
            st.warning("No matching songs found. Please try a different input.")

        # Number of recommendations
        k = st.selectbox("🔢 How many recommendations do you want?", [5, 10, 15, 20], index=1)

        # Trigger recommendations
        if selected_label and st.button("🚀 Get Recommendations"):
            selected_song_name = label_to_name_map[selected_label]
            logger.info(f'User selected "{selected_song_name}" with k={k}.')
            self.handle_recommendations(
                song_name=selected_song_name, k=k, cleaned_data=cleaned_data
            )

        logger.info("App ran successfully.")


if __name__ == "__main__":
    app = SpotifyRecommenderApp(
        cleaned_data_path=CLEANED_SONGS_DATA, transformed_data_path=TRANSFORMED_SONGS_DATA_CBF
    )
    app.run()
