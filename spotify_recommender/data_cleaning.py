import pandas as pd

from loguru import logger
from tqdm.auto import tqdm
from pathlib import Path

from spotify_recommender.config import (
    RAW_MUSIC_DATA,
    CLEANED_MUSIC_DATA,
    DATA_CLEANING_DROP_DUPLICATES_SUBSET,
    DATA_CLEANING_COLS_TO_DROP,
    DATA_CLEANING_FILL_NA_VALS_DICT,
    DATA_CLEANING_COLS_TO_LOWERCASE,
)


class DataCleaner:
    def __init__(self, raw_data_path: Path) -> None:
        """Initiates a `DataCleaner` object.

        Args:
            raw_data_path (Path): Path of the raw data.
        """
        logger.info("Instantiating a `DataCleaner` object...")
        self.raw_data_path = raw_data_path
        logger.info("`DataCleaner` object successfully instantiated.")

    def load_data(self) -> pd.DataFrame:
        """Loads the raw data.

        Returns:
            pd.DataFrame: Raw data.
        """
        try:
            logger.info("Running the method `load_data`...")
            logger.info(f"Loading data from '{self.raw_data_path}'...")
            data = pd.read_csv(self.raw_data_path)
            logger.info("Data loaded successfully.")
            return data
        except FileNotFoundError:
            logger.error(f"File '{self.raw_data_path}' not found.")
            raise
        except pd.errors.EmptyDataError:
            logger.error("CSV file is empty.")
            raise
        except pd.errors.ParserError:
            logger.error("Error parsing the CSV file.")
            raise

    def drop_duplicates(self, data: pd.DataFrame, subset: list) -> pd.DataFrame:
        """Drops duplicate rows from the data based on a given subset of columns.

        Args:
            data (pd.DataFrame): Data with duplicates.
            subset (list): Subset of columns.

        Returns:
            pd.DataFrame: Data after duplicates are dropped.
        """
        try:
            logger.info("Running the method `drop_duplicates`...")
            logger.info(f"Removing duplicates based on columns: {subset}...")
            data = data.drop_duplicates(subset=subset).reset_index(drop=True)
            logger.info("Duplicates dropped successfully.")
            return data
        except KeyError as e:
            logger.error(f"KeyError in `drop_duplicates`: {e}.")
            raise

    def drop_unnecessary_cols(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Drops a list of unnecessary columns from the data.

        Args:
            data (pd.DataFrame): Data with unnecessary columns.
            columns (list): Unnecessary columns.

        Returns:
            pd.DataFrame: Data after the columns are dropped.
        """
        try:
            logger.info("Running the method `drop_unnecessary_cols`...")
            logger.info(f"Dropping columns: {columns}...")
            data = data.drop(columns=columns).reset_index(drop=True)
            logger.info("Columns dropped successfully.")
            return data
        except KeyError as e:
            logger.error(f"KeyError in `drop_unnecessary_cols`: {e}.")
            raise

    def fill_missing_values(self, data: pd.DataFrame, fill_na_vals_dict: dict) -> pd.DataFrame:
        """Fills missing values in columns specified as a key-value pair.

        Args:
            data (pd.DataFrame): Data with missing values.
            fill_na_vals_dict (dict): Key-value pairs of columns and the corresponding value to fill.

        Returns:
            pd.DataFrame: Data after missing values are filled.
        """
        try:
            logger.info("Running the method `fill_missing_values`...")
            logger.info(f"Filling missing values using the dictionary: {fill_na_vals_dict}...")
            data = data.fillna(fill_na_vals_dict)
            logger.info("Missing values filled successfully.")
            return data
        except Exception as e:
            logger.error(f"Unexpected error in `fill_missing_values`: {e}.")
            raise

    def convert_column_entries_to_lowercase(
        self, data: pd.DataFrame, columns: list
    ) -> pd.DataFrame:
        """Converts entries in a list of columns to lowercase.

        Args:
            data (pd.DataFrame): Data before the conversion.
            columns (list): List of columns that are to be converted.

        Returns:
            pd.DataFrame: Data after conversion.
        """
        try:
            logger.info("Running the method `convert_column_entries_to_lowercase`...")
            logger.info(f"Converting the entries in the columns {columns} to lowercase...")
            for column in tqdm(columns, desc="Converting column entries to lowercase", unit="col"):
                data[column] = data[column].astype(str).str.lower()
            return data
        except KeyError as e:
            logger.error(f"KeyError in `convert_column_entries_to_lowercase`: {e}.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in `convert_column_entries_to_lowercase`: {e}.")
            raise

    def clean_data(
        self,
        drop_duplicates_subset: list,
        cols_to_drop: list,
        fill_na_vals_dict: dict,
        cols_to_lowercase: list,
    ) -> pd.DataFrame:
        """The orchestrator method that performs data cleaning.

        Args:
            drop_duplicates_subset (list): Subset of columns based on which duplicates are to be dropped.
            cols_to_drop (list): List of columns to drop.
            fill_na_vals_dict (dict): Key-value pairs of columns and the corresponding fill value.
            cols_to_lowercase (list): List of columns whose entries are to be converted to lowercase.

        Returns:
            pd.DataFrame: Cleaned data.
        """
        try:
            logger.info("Running the method `clean_data`...")
            with tqdm(total=4, desc="Data Cleaning Process", unit="step") as pbar:
                data = self.load_data()
                pbar.update(1)

                data = self.drop_duplicates(data=data, subset=drop_duplicates_subset)
                pbar.update(1)

                data = self.drop_unnecessary_cols(data=data, columns=cols_to_drop)
                pbar.update(1)

                data = self.fill_missing_values(data=data, fill_na_vals_dict=fill_na_vals_dict)
                pbar.update(1)

                data = self.convert_column_entries_to_lowercase(
                    data=data, columns=cols_to_lowercase
                )
                pbar.update(1)

            logger.info("The method `clean_data` ran successfully.")
            return data
        except Exception as e:
            logger.error(f"Unexpected error in `clean_data`: {e}.")
            raise


if __name__ == "__main__":
    try:
        logger.info("Starting data cleaning process...")

        # Initialize DataCleaner object
        data_cleaner = DataCleaner(raw_data_path=RAW_MUSIC_DATA)

        # Clean the data
        cleaned_data = data_cleaner.clean_data(
            drop_duplicates_subset=DATA_CLEANING_DROP_DUPLICATES_SUBSET,
            cols_to_drop=DATA_CLEANING_COLS_TO_DROP,
            fill_na_vals_dict=DATA_CLEANING_FILL_NA_VALS_DICT,
            cols_to_lowercase=DATA_CLEANING_COLS_TO_LOWERCASE,
        )

        # Save cleaned data
        logger.info(f"Saving the cleaned data to '{CLEANED_MUSIC_DATA}'...")
        cleaned_data.to_csv(CLEANED_MUSIC_DATA, index=False)
        logger.info("Data saved successfully.")

        logger.info("Data cleaning process completed successfully!")
    except Exception as e:
        logger.critical(f"Data cleaning process failed: {e}.")
