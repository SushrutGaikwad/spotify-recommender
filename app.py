import pandas as pd
import streamlit as st

from pathlib import Path
from loguru import logger
from tqdm.auto import tqdm
from scipy.sparse import load_npz, spmatrix

# CBF
from spotify_recommender.content_based_filtering.recommendation import (
    Recommender as CBF_Recommender,
)

# CF
from spotify_recommender.collaborative_filtering.recommendation import (
    Recommender as CF_Recommender,
)

# Hybrid
from spotify_recommender.hybrid_recommendation.recommendation import (
    Recommender as Hybrid_Recommender,
)

from spotify_recommender.config import (
    # Content-based filtering
    CLEANED_SONGS_DATA,
    TRANSFORMED_SONGS_DATA_CBF,
    # Collaborative filtering
    COLLAB_FILTERED_SONGS_DATA,
    COLLAB_TRACK_IDS_NPY,
    COLLAB_INTERACTION_MATRIX_NPZ,
    # Hybrid
    HYBRID_FILTERED_SONGS_SORTED,
    HYBRID_TRANSFORMED_CBF_NPZ,
    HYBRID_WEIGHT_CBF,
    HYBRID_WEIGHT_CF,
)


class SpotifyRecommenderApp:
    def __init__(self, cleaned_data_path: Path, transformed_data_path: Path) -> None:
        """Initiates a `SpotifyRecommenderApp` object.

        Args:
            cleaned_data_path (Path): Path of the cleaned data for content-based filtering.
            transformed_data_path (Path): Path of the transformed data for content-based filtering.
        """
        logger.info("Instantiating a `SpotifyRecommenderApp` object...")
        self.cleaned_data_path = cleaned_data_path
        self.transformed_data_path = transformed_data_path

        # Content-based filtering recommender
        self.recommender_cbf = CBF_Recommender(
            cleaned_data_path=self.cleaned_data_path,
            transformed_data_path=self.transformed_data_path,
        )

        # Collaborative filtering recommender
        self.recommender_cf = CF_Recommender(
            filtered_songs_path=COLLAB_FILTERED_SONGS_DATA,
            track_ids_path=COLLAB_TRACK_IDS_NPY,
            interaction_matrix_path=COLLAB_INTERACTION_MATRIX_NPZ,
        )

        # Hybrid recommender
        self.recommender_hybrid = Hybrid_Recommender(
            filtered_songs_path=HYBRID_FILTERED_SONGS_SORTED,
            hybrid_cbf_features_path=HYBRID_TRANSFORMED_CBF_NPZ,
            track_ids_path=COLLAB_TRACK_IDS_NPY,
            interaction_matrix_path=COLLAB_INTERACTION_MATRIX_NPZ,
        )

        logger.info("`SpotifyRecommenderApp` object successfully instantiated.")

    def load_cleaned_data(self) -> pd.DataFrame | None:
        """Loads the cleaned data.

        Returns:
            pd.DataFrame | None: Cleaned data.
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
            logger.error(f"Unexpected error in `load_cleaned_data`: {e}.")
            st.error("An error occurred while loading the data.")

    def load_transformed_data(self) -> spmatrix | None:
        """Loads the transformed data (content-based).

        Returns:
            spmatrix | None: Transformed data.
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
            logger.error(f"Unexpected error in `load_transformed_data`: {e}.")
            st.error("An error occurred while loading the data.")

    def load_cf_catalog(self) -> pd.DataFrame | None:
        """Loads the filtered catalog used by collaborative filtering.

        Returns:
            pd.DataFrame | None: Filtered songs catalog for collaborative filtering.
        """
        try:
            logger.info("Running the method `load_cf_catalog`...")
            logger.info(f"Loading CF filtered catalog from '{COLLAB_FILTERED_SONGS_DATA}'...")
            cf_catalog = pd.read_csv(COLLAB_FILTERED_SONGS_DATA)
            return cf_catalog
        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError: {e}.")
            st.error(f"Error: Missing collaborative filtering files! {e}.")
        except Exception as e:
            logger.error(f"Unexpected error in `load_cf_catalog`: {e}.")
            st.error("An error occurred while loading collaborative filtering data.")

    def load_hybrid_catalog(self) -> pd.DataFrame | None:
        """Loads the hybrid recommendation catalog (sorted CF universe)."""
        try:
            logger.info("Running the method `load_hybrid_catalog`...")
            logger.info(f"Loading Hybrid catalog from '{HYBRID_FILTERED_SONGS_SORTED}'...")
            hybrid_catalog = pd.read_csv(HYBRID_FILTERED_SONGS_SORTED)
            return hybrid_catalog
        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError: {e}.")
            st.error(f"Error: Missing hybrid recommendation files! {e}.")
        except Exception as e:
            logger.error(f"Unexpected error in `load_hybrid_catalog`: {e}.")
            st.error("An error occurred while loading hybrid recommendation data.")

    def get_cbf_recommendations(
        self, song_name: str, k: int, cleaned_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Generates content-based filtering recommendations.

        Args:
            song_name (str): Name of the song to generate recommendations from.
            k (int): The number of recommendations to generate.
            cleaned_data (pd.DataFrame): The cleaned data.

        Returns:
            pd.DataFrame: Recommendations.
        """
        try:
            logger.info("Running the method `get_cbf_recommendations`...")
            logger.info(f'Fetching k={k} CBF recommendations for "{song_name}"...')
            song_name_lower = song_name.lower()

            if (cleaned_data["name"] == song_name_lower).any():
                with tqdm(total=1, desc="Generating CBF Recommendations", unit="step") as pbar:
                    recommendations = self.recommender_cbf.recommend(song_name=song_name, k=k)
                    pbar.update(1)
                logger.info(f"Successfully generated {k} CBF recommendations.")
                return recommendations
            else:
                logger.warning(f'The song "{song_name}" is not found in the cleaned dataset.')
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error in `get_cbf_recommendations`: {e}.")
            return pd.DataFrame()

    def get_cf_recommendations(self, song_name: str, artist_name: str, k: int) -> pd.DataFrame:
        """Generates collaborative filtering recommendations.

        Args:
            song_name (str): Name of the seed song.
            artist_name (str): Artist of the seed song.
            k (int): The number of recommendations to generate.

        Returns:
            pd.DataFrame: Recommendations.
        """
        try:
            logger.info("Running the method `get_cf_recommendations`...")
            logger.info(
                f'Fetching k={k} CF recommendations for "{song_name}" by "{artist_name}"...'
            )

            with tqdm(total=1, desc="Generating CF Recommendations", unit="step") as pbar:
                recommendations = self.recommender_cf.recommend(
                    song_name=song_name, artist_name=artist_name, k=k
                )
                pbar.update(1)
            logger.info(f"Successfully generated {k} CF recommendations.")
            return recommendations
        except ValueError as e:
            logger.warning(str(e))
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error in `get_cf_recommendations`: {e}.")
            return pd.DataFrame()

    def _build_display_labels(self, df: pd.DataFrame) -> tuple[list[str], dict]:
        """Builds display labels and a mapping for selection.

        Args:
            df (pd.DataFrame): DataFrame with 'name' and 'artist' columns.

        Returns:
            tuple[list[str], dict]: (labels, label->(name, artist)) mapping.
        """
        try:
            logger.info("Running the helper `_build_display_labels`...")
            display_labels, label_map = [], {}
            for _, row in df.iterrows():
                label = f'{row["name"].title()} — {row["artist"].title()}'
                display_labels.append(label)
                # Save lowercase (as data is lowercased by cleaner)
                label_map[label] = (row["name"], row["artist"])
            return display_labels, label_map
        except Exception as e:
            logger.error(f"Unexpected error in `_build_display_labels`: {e}.")
            return [], {}

    def display_recommendations(
        self,
        seed_name: str,
        seed_artist: str,
        seed_preview_url: str | None,
        recommendations: pd.DataFrame,
    ) -> None:
        """Displays the seed track first and then the recommendations.

        Args:
            seed_name (str): Seed song name (title-cased for display is handled here).
            seed_artist (str): Seed artist name.
            seed_preview_url (str | None): Preview URL for the seed (can be None).
            recommendations (pd.DataFrame): Recommendations (seed excluded).
        """
        try:
            logger.info("Running the method `display_recommendations`...")

            # Seed / Currently Playing
            st.markdown("### Currently Playing")
            st.markdown(f'##### "{seed_name.title()}" by "{seed_artist.title()}":')
            if seed_preview_url:
                st.audio(seed_preview_url)
            st.write("---")

            if recommendations.empty:
                st.info("No recommendations to display.")
                return

            # Recommendations (start numbering at 1)
            with tqdm(
                total=len(recommendations), desc="Displaying Recommendations", unit="song"
            ) as pbar:
                for rank, recommendation in enumerate(
                    recommendations.itertuples(index=False), start=1
                ):
                    if rank == 1:
                        st.markdown("### You may also like")
                    rec_song = getattr(recommendation, "name").title()
                    rec_artist = getattr(recommendation, "artist").title()
                    st.markdown(f'##### {rank}. "{rec_song}" by "{rec_artist}":')
                    preview = getattr(recommendation, "spotify_preview_url", None)
                    if preview:
                        st.audio(preview)
                    st.write("---")
                    pbar.update(1)

            logger.info("Recommendations displayed successfully.")
        except Exception as e:
            logger.error(f"Unexpected error in `display_recommendations`: {e}.")
            st.error("An error occurred while displaying recommendations.")

    def handle_cbf(self, cleaned_data: pd.DataFrame) -> None:
        """Handles the CBF UI flow and displays recommendations.

        Args:
            cleaned_data (pd.DataFrame): Cleaned data used for CBF searching.
        """
        try:
            logger.info("Running the method `handle_cbf`...")

            typed_input = st.text_input("Type a song name and press enter/return:")
            filtered_df = (
                cleaned_data[
                    cleaned_data["name"].str.contains(typed_input.lower(), na=False, regex=False)
                ]
                if typed_input
                else pd.DataFrame()
            )

            if typed_input and filtered_df.empty:
                st.warning("No matching songs found. Please try a different input.")
                return

            labels, label_map = self._build_display_labels(filtered_df)
            selected_label = (
                st.selectbox("Matching songs found. Pick yours:", sorted(labels))
                if labels
                else None
            )

            k = st.selectbox("How many recommendations do you want?", [5, 10, 15, 20], index=1)

            if selected_label and st.button("Get Recommendations!"):
                selected_song_name, selected_artist_name = label_map[selected_label]
                logger.info(f'CBF selected "{selected_song_name}" with k={k}.')

                recs = self.get_cbf_recommendations(
                    song_name=selected_song_name, k=k, cleaned_data=cleaned_data
                )

                seed_row = cleaned_data.loc[
                    (cleaned_data["name"] == selected_song_name)
                    & (cleaned_data["artist"] == selected_artist_name)
                ]
                seed_preview = (
                    seed_row["spotify_preview_url"].iloc[0] if not seed_row.empty else None
                )

                if recs.empty:
                    st.error(
                        f"""Sorry, we couldn't find find recommendations for "{selected_song_name}" in our database."""
                    )
                else:
                    st.markdown("## Recommendations")
                    self.display_recommendations(
                        seed_name=selected_song_name,
                        seed_artist=selected_artist_name,
                        seed_preview_url=seed_preview,
                        recommendations=recs,
                    )

        except Exception as e:
            logger.error(f"Unexpected error in `handle_cbf`: {e}.")
            st.error(
                "Something went wrong while generating CBF recommendations. Please try again."
            )

    def handle_cf(self, cf_catalog: pd.DataFrame) -> None:
        """Handles the CF UI flow and displays recommendations.

        Args:
            cf_catalog (pd.DataFrame): Filtered catalog used for CF searching.
        """
        try:
            logger.info("Running the method `handle_cf`...")

            typed_input = st.text_input("Type a song name and press enter/return:")
            filtered_df = (
                cf_catalog[
                    cf_catalog["name"].str.contains(typed_input.lower(), na=False, regex=False)
                ]
                if typed_input
                else pd.DataFrame()
            )

            if typed_input and filtered_df.empty:
                st.warning("No matching songs found. Please try a different input.")
                return

            labels, label_map = self._build_display_labels(filtered_df)
            selected_label = (
                st.selectbox("Matching songs found. Pick yours:", sorted(labels))
                if labels
                else None
            )

            k = st.selectbox("How many recommendations do you want?", [5, 10, 15, 20], index=1)

            if selected_label and st.button("Get Recommendations!"):
                selected_song_name, selected_artist_name = label_map[selected_label]
                logger.info(
                    f'CF selected "{selected_song_name}" by "{selected_artist_name}" with k={k}.'
                )

                recs = self.get_cf_recommendations(
                    song_name=selected_song_name, artist_name=selected_artist_name, k=k
                )

                seed_row = cf_catalog.loc[
                    (cf_catalog["name"] == selected_song_name)
                    & (cf_catalog["artist"] == selected_artist_name)
                ]
                seed_preview = (
                    seed_row["spotify_preview_url"].iloc[0] if not seed_row.empty else None
                )

                if recs.empty:
                    st.error(
                        f"""Sorry, we couldn't find "{selected_song_name}" by "{selected_artist_name}" in the CF index."""
                    )
                else:
                    st.markdown("## Recommendations")
                    self.display_recommendations(
                        seed_name=selected_song_name,
                        seed_artist=selected_artist_name,
                        seed_preview_url=seed_preview,
                        recommendations=recs,
                    )

        except Exception as e:
            logger.error(f"Unexpected error in `handle_cf`: {e}.")
            st.error("Something went wrong while generating CF recommendations. Please try again.")

    def handle_hybrid(self, hybrid_catalog: pd.DataFrame) -> None:
        """Handles the Hybrid UI flow and displays recommendations.

        Args:
            hybrid_catalog (pd.DataFrame): Sorted CF universe used by Hybrid searching.
        """
        try:
            logger.info("Running the method `handle_hybrid`...")

            typed_input = st.text_input("Type a song name and press enter/return:")
            filtered_df = (
                hybrid_catalog[
                    hybrid_catalog["name"].str.contains(typed_input.lower(), na=False, regex=False)
                ]
                if typed_input
                else pd.DataFrame()
            )

            if typed_input and filtered_df.empty:
                st.warning("No matching songs found. Please try a different input.")
                return

            labels, label_map = self._build_display_labels(filtered_df)
            selected_label = (
                st.selectbox("Matching songs found. Pick yours:", sorted(labels))
                if labels
                else None
            )

            k = st.selectbox("How many recommendations do you want?", [5, 10, 15, 20], index=1)

            # Single slider for CBF weight; CF weight auto-adjusts to sum to 1.
            default_w_cbf = min(max(HYBRID_WEIGHT_CBF, 0.0), 1.0)
            w_cbf = st.slider(
                r"Select a $w_{\text{cbf}}$ value (higher ⇒ more similar songs; lower ⇒ more songs from people that have listened to the same song):",
                min_value=0.0,
                max_value=1.0,
                value=default_w_cbf,
                step=0.01,
            )
            w_cf = 1.0 - w_cbf
            st.latex(
                rf"\text{{Using weights: }} w_{{\text{{cbf}}}} = {w_cbf:.2f},\quad "
                rf"w_{{\text{{cf}}}} = {w_cf:.2f} \quad (\text{{sum}} = 1.00)"
            )

            if selected_label and st.button("Get Recommendations!"):
                selected_song_name, selected_artist_name = label_map[selected_label]
                logger.info(
                    f'Hybrid selected "{selected_song_name}" by "{selected_artist_name}" '
                    f"with k={k}, w_cbf={w_cbf:.2f}, w_cf={w_cf:.2f}."
                )

                with tqdm(total=1, desc="Generating Hybrid Recommendations", unit="step") as pbar:
                    recs = self.recommender_hybrid.recommend(
                        song_name=selected_song_name,
                        artist_name=selected_artist_name,
                        k=k,
                        weight_cbf=w_cbf,
                        weight_cf=w_cf,
                    )
                    pbar.update(1)

                seed_row = hybrid_catalog.loc[
                    (hybrid_catalog["name"] == selected_song_name)
                    & (hybrid_catalog["artist"] == selected_artist_name)
                ]
                seed_preview = (
                    seed_row["spotify_preview_url"].iloc[0] if not seed_row.empty else None
                )

                if recs.empty:
                    st.error(
                        f"""Sorry, we couldn't find "{selected_song_name}" by "{selected_artist_name}" in the Hybrid index."""
                    )
                else:
                    st.markdown("## Recommendations")
                    self.display_recommendations(
                        seed_name=selected_song_name,
                        seed_artist=selected_artist_name,
                        seed_preview_url=seed_preview,
                        recommendations=recs,
                    )

        except Exception as e:
            logger.error(f"Unexpected error in `handle_hybrid`: {e}.")
            st.error(
                "Something went wrong while generating hybrid recommendations. Please try again."
            )

    def run(self) -> None:
        """The orchestrator method that runs the app."""
        logger.info("Running the method `run`...")
        logger.info("Running the app...")

        st.title("Welcome to the Spotify Song Recommender!")
        st.write("Start typing a song name to get suggestions. We'll recommend similar songs.")

        cleaned_data = self.load_cleaned_data()
        if (
            cleaned_data is None
            or "name" not in cleaned_data.columns
            or "artist" not in cleaned_data.columns
        ):
            st.error("The cleaned dataset must contain 'name' and 'artist' columns.")
            return

        cleaned_data = cleaned_data.dropna(subset=["name", "artist"])
        cleaned_data["name"] = cleaned_data["name"].astype(str)
        cleaned_data["artist"] = cleaned_data["artist"].astype(str)

        cf_catalog = self.load_cf_catalog()
        if cf_catalog is not None:
            cf_catalog = cf_catalog.dropna(subset=["name", "artist"])
            cf_catalog["name"] = cf_catalog["name"].astype(str)
            cf_catalog["artist"] = cf_catalog["artist"].astype(str)

        hybrid_catalog = self.load_hybrid_catalog()
        if hybrid_catalog is not None:
            hybrid_catalog = hybrid_catalog.dropna(subset=["name", "artist"])
            hybrid_catalog["name"] = hybrid_catalog["name"].astype(str)
            hybrid_catalog["artist"] = hybrid_catalog["artist"].astype(str)

        filtering_type = st.selectbox(
            "Select the recommendation strategy:",
            ["Content-Based Filtering", "Collaborative Filtering", "Hybrid Recommendation"],
            index=2,
        )

        if filtering_type == "Content-Based Filtering":
            self.handle_cbf(cleaned_data=cleaned_data)
        elif filtering_type == "Collaborative Filtering":
            if cf_catalog is None or cf_catalog.empty:
                st.error(
                    "Collaborative filtering artifacts are missing. "
                    "Please run the CF pipeline to generate the filtered catalog and interaction matrix."
                )
            else:
                self.handle_cf(cf_catalog=cf_catalog)
        else:  # Hybrid Recommendation
            if hybrid_catalog is None or hybrid_catalog.empty:
                st.error(
                    "Hybrid recommendation artifacts are missing. "
                    "Please run the hybrid pipeline to generate the sorted catalog and hybrid features."
                )
            else:
                self.handle_hybrid(hybrid_catalog=hybrid_catalog)

        logger.info("App ran successfully.")


if __name__ == "__main__":
    app = SpotifyRecommenderApp(
        cleaned_data_path=CLEANED_SONGS_DATA,
        transformed_data_path=TRANSFORMED_SONGS_DATA_CBF,
    )
    app.run()
