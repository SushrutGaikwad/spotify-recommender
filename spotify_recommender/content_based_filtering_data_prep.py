import pandas as pd

from loguru import logger
from tqdm.auto import tqdm
from pathlib import Path

from spotify_recommender.config import (
    CLEANED_MUSIC_DATA,
    CLEANED_MUSIC_DATA_CBF,
    CONTENT_BASED_FILTERING_DATA_PREP_COLS_TO_DROP,
)


class ContentBasedFilteringDataPrepper:
    def __init__(self, cleaned_data_path: Path) -> None:
        """Initiates a `ContentBasedFilteringDataPrepper` object.

        Args:
            cleaned_data_path (Path): Path of the cleaned data.
        """
        logger.info("Instantiating a `ContentBasedFilteringDataPrepper` object...")
        self.cleaned_data_path = cleaned_data_path
        logger.info("`ContentBasedFilteringDataPrepper` object successfully instantiated.")

    def load_data(self) -> pd.DataFrame:
        """Loads the cleaned data.

        Returns:
            pd.DataFrame: Cleaned data.
        """
        try:
            logger.info("Running the method `load_data`...")
            logger.info(f"Loading data from '{self.cleaned_data_path}'...")
            data = pd.read_csv(self.cleaned_data_path)
            logger.info(
                f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns."
            )
            return data
        except FileNotFoundError:
            logger.error(f"File '{self.cleaned_data_path}' not found.")
            raise
        except pd.errors.EmptyDataError:
            logger.error("CSV file is empty.")
            raise
        except pd.errors.ParserError:
            logger.error("Error parsing the CSV file.")
            raise

    def drop_columns(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Drops unwanted columns from the data.

        Args:
            data (pd.DataFrame): Data.
            columns (list): Columns to drop.

        Returns:
            pd.DataFrame: Data after columns are dropped.
        """
        try:
            logger.info("Running the method `drop_columns`...")
            logger.info(f"Dropping columns: {columns}...")
            data = data.drop(columns=columns)
            logger.info(f"Columns dropped successfully. New shape: {data.shape}.")
            return data
        except KeyError as e:
            logger.error(f"KeyError in `drop_columns`: {e}.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in `drop_columns`: {e}.")
            raise

    def prepare_data(self, cols_to_drop: list) -> pd.DataFrame:
        """The orchestrator method that prepares data for content-based filtering.

        Args:
            cols_to_drop (list): Unwanted columns to drop.

        Returns:
            pd.DataFrame: Prepared data.
        """
        try:
            logger.info("Running the method `prepare_data`...")
            with tqdm(
                total=2, desc="Data Preparation Process for Content-based Filtering", unit="step"
            ) as pbar:
                data = self.load_data()
                pbar.update(1)

                data = self.drop_columns(data=data, columns=cols_to_drop)
                pbar.update(1)

            logger.info("The method `prepare_data` ran successfully.")
            return data
        except Exception as e:
            logger.error(f"Unexpected error in `prepare_data`: {e}.")
            raise


if __name__ == "__main__":
    try:
        logger.info("Starting data preparation process for content-based filtering...")

        # Initialize ContentBasedFilteringDataPrepper object
        data_prepper = ContentBasedFilteringDataPrepper(cleaned_data_path=CLEANED_MUSIC_DATA)

        # Prepare the data
        cbf_data = data_prepper.prepare_data(
            cols_to_drop=CONTENT_BASED_FILTERING_DATA_PREP_COLS_TO_DROP
        )

        # Save the prepared data
        logger.info(f"Saving the prepared data to '{CLEANED_MUSIC_DATA_CBF}'...")
        cbf_data.to_csv(CLEANED_MUSIC_DATA_CBF)
        logger.info("Data saved successfully.")

        logger.info("Data preparation process for content-based filtering completed successfully!")
    except Exception as e:
        logger.critical(f"Data preparation for content-based filtering process failed: {e}.")
