import joblib

import pandas as pd

from typing import Tuple
from pathlib import Path
from loguru import logger
from tqdm.auto import tqdm
from scipy.sparse import save_npz
from category_encoders.count import CountEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

from spotify_recommender.config import (
    CLEANED_MUSIC_DATA_CBF,
    TRANSFORMED_MUSIC_DATA_CBF,
    TRAINED_TRANSFORMER,
    CONTENT_BASED_FILTERING_DATA_TRANS_TFIDF_MAX_FEATURES,
    CONTENT_BASED_FILTERING_DATA_TRANS_FREQUENCY_ENCODE_COLS,
    CONTENT_BASED_FILTERING_DATA_TRANS_OHE_COLS,
    CONTENT_BASED_FILTERING_DATA_TRANS_TFIDF_COL,
    CONTENT_BASED_FILTERING_DATA_TRANS_STD_SCALER_COLS,
    CONTENT_BASED_FILTERING_DATA_TRANS_MIN_MAX_SCALER_COLS,
)


class ContentBasedFilteringDataTransformer:
    def __init__(self, prepared_data_path: Path) -> None:
        """Initiates a `ContentBasedFilteringDataTransformer` object.

        Args:
            prepared_data_path (Path): Path of the prepared data.
        """
        logger.info("Instantiating a `ContentBasedFilteringDataTransformer` object...")
        self.prepared_data_path = prepared_data_path
        logger.info("`ContentBasedFilteringDataTransformer` object successfully instantiated.")

    def load_data(self) -> pd.DataFrame:
        """Loads the prepared data.

        Returns:
            pd.DataFrame: Prepared data.
        """
        try:
            logger.info("Running the method `load_data`...")
            logger.info(f"Loading data from '{self.prepared_data_path}'...")
            data = pd.read_csv(self.prepared_data_path)
            logger.info(
                f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns."
            )
            return data
        except FileNotFoundError:
            logger.error(f"File '{self.prepared_data_path}' not found.")
            raise
        except pd.errors.EmptyDataError:
            logger.error("CSV file is empty.")
            raise
        except pd.errors.ParserError:
            logger.error("Error parsing the CSV file.")
            raise

    def create_transformer(self) -> ColumnTransformer:
        """Creates a transformer.

        Returns:
            ColumnTransformer: The transformer.
        """
        try:
            logger.info("Running the method `create_transformer`...")
            logger.info("Creating a transformer...")
            transformer = ColumnTransformer(
                transformers=[
                    (
                        "frequency_encoder",
                        CountEncoder(normalize=True, return_df=True),
                        CONTENT_BASED_FILTERING_DATA_TRANS_FREQUENCY_ENCODE_COLS,
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                        CONTENT_BASED_FILTERING_DATA_TRANS_OHE_COLS,
                    ),
                    (
                        "tf-idf",
                        TfidfVectorizer(
                            max_features=CONTENT_BASED_FILTERING_DATA_TRANS_TFIDF_MAX_FEATURES
                        ),
                        CONTENT_BASED_FILTERING_DATA_TRANS_TFIDF_COL,
                    ),
                    (
                        "standard_scaler",
                        StandardScaler(),
                        CONTENT_BASED_FILTERING_DATA_TRANS_STD_SCALER_COLS,
                    ),
                    (
                        "min_max_scaler",
                        MinMaxScaler(),
                        CONTENT_BASED_FILTERING_DATA_TRANS_MIN_MAX_SCALER_COLS,
                    ),
                ],
                remainder="passthrough",
                n_jobs=-1,
                force_int_remainder_cols=False,
            )
            logger.info("Transformer created successfully.")
            return transformer
        except Exception as e:
            logger.error(f"Error in `create_transformer`: {e}.")
            raise

    def train_transformer(
        self, data: pd.DataFrame, transformer: ColumnTransformer
    ) -> ColumnTransformer:
        """Trains a given transformer on the given data.

        Args:
            data (pd.DataFrame): Data.
            transformer (ColumnTransformer): Transformer.

        Returns:
            ColumnTransformer: Trained transformer.
        """
        try:
            logger.info("Running the method `train_transformer`...")
            freq_encode_cols = CONTENT_BASED_FILTERING_DATA_TRANS_FREQUENCY_ENCODE_COLS
            for col in freq_encode_cols:
                if col in data.columns and data[col].dtype not in ["object", "category"]:
                    logger.warning(
                        f"Column '{col}' is not categorical. Converting to 'category' type."
                    )
                    data[col] = data[col].astype("category")
            logger.info("Training the transformer...")
            with tqdm(total=1, desc="Training Transformer", unit="step") as pbar:
                trained_transformer = transformer.fit(data)
                pbar.update(1)
            logger.info("Transformer trained successfully.")
            return trained_transformer
        except ValueError as e:
            logger.error(f"ValueError in `train_transformer`: {e}.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in `train_transformer`: {e}.")
            raise

    def transform_data(
        self, data: pd.DataFrame, trained_transformer: ColumnTransformer
    ) -> pd.DataFrame:
        """Transforms the given data using a trained transformer.

        Args:
            data (pd.DataFrame): Data.
            trained_transformer (ColumnTransformer): The trained transformer.

        Returns:
            pd.DataFrame: Data after transformation.
        """
        try:
            logger.info("Running the method `transform_data`...")
            logger.info("Transforming the data...")
            with tqdm(total=1, desc="Transforming Data", unit="step") as pbar:
                transformed_data = trained_transformer.transform(data)
                pbar.update(1)
            logger.info("Data transformed successfully.")
            return transformed_data
        except ValueError as e:
            logger.error(f"ValueError in `transform_data`: {e}.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in `transform_data`: {e}.")
            raise

    def transform_data_for_content_based_filtering(self) -> Tuple[ColumnTransformer, pd.DataFrame]:
        """The orchestrator method that transforms the data for content-based filtering.

        Returns:
            Tuple[ColumnTransformer, pd.DataFrame]: The trained transformer and the transformed data.
        """
        try:
            logger.info("Running the method `transform_data_for_content_based_filtering`...")
            with tqdm(
                total=3, desc="Content-Based Filtering Data Transformation Process", unit="step"
            ) as pbar:
                data = self.load_data()
                pbar.update(1)

                transformer = self.create_transformer()
                trained_transformer = self.train_transformer(data=data, transformer=transformer)
                pbar.update(1)

                transformed_data = self.transform_data(
                    data=data, trained_transformer=trained_transformer
                )
                pbar.update(1)

            logger.info(
                "The method `transform_data_for_content_based_filtering` ran successfully."
            )
            return trained_transformer, transformed_data
        except Exception as e:
            logger.error(f"Unexpected error in `transform_data_for_content_based_filtering`: {e}.")
            raise


if __name__ == "__main__":
    try:
        logger.info("Starting data transformation process for content-based filtering...")

        # Initialize ContentBasedFilteringDataTransformer object
        data_transformer = ContentBasedFilteringDataTransformer(
            prepared_data_path=CLEANED_MUSIC_DATA_CBF
        )

        # Perform data transformation for content-based filtering
        trained_transformer, transformed_data = (
            data_transformer.transform_data_for_content_based_filtering()
        )

        # Save the trained transformer
        logger.info(f"Saving trained transformer to '{TRAINED_TRANSFORMER}'...")
        joblib.dump(trained_transformer, TRAINED_TRANSFORMER)
        logger.info("Trained transformer saved successfully.")

        # Save the transformed data
        logger.info(f"Saving transformed data to '{TRANSFORMED_MUSIC_DATA_CBF}'...")
        save_npz(TRANSFORMED_MUSIC_DATA_CBF, transformed_data)
        logger.info("Data saved successfully.")

        logger.info(
            "Data transformation process for content-based filtering completed successfully!"
        )
    except Exception as e:
        logger.critical(f"Data transformation process for content-based filtering failed: {e}.")
