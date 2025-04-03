import os
import logging
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import requests
from io import StringIO

# Configure logging
LOG_DIR = "05_ml_pipeline_dvc_s3/logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOGGER = logging.getLogger("data_ingestion")
LOGGER.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(LOG_DIR, "data_ingestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

LOGGER.addHandler(console_handler)
LOGGER.addHandler(file_handler)


def load_config(config_path: str) -> dict:
    """Load configuration parameters from a YAML file."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        LOGGER.debug("Loaded configuration from %s", config_path)
        return config
    except FileNotFoundError:
        LOGGER.error("Configuration file not found: %s", config_path)
        raise
    except yaml.YAMLError as err:
        LOGGER.error("YAML parsing error: %s", err)
        raise
    except Exception as err:
        LOGGER.error("Unexpected error while loading configuration: %s", err)
        raise


def fetch_data(csv_url: str) -> pd.DataFrame:
    """Fetch dataset from the given CSV URL."""
    try:
        response = requests.get(csv_url, verify=False) 
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        LOGGER.debug("Data successfully fetched from %s", csv_url)
        return df
    except pd.errors.ParserError as err:
        LOGGER.error("CSV parsing error: %s", err)
        raise
    except Exception as err:
        LOGGER.error("Unexpected error while fetching data: %s", err)
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the dataset."""
    try:
        df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        LOGGER.debug("Data preprocessing completed successfully")
        return df
    except KeyError as err:
        LOGGER.error("Missing expected column(s) in dataset: %s", err)
        raise
    except Exception as err:
        LOGGER.error("Unexpected error during data preprocessing: %s", err)
        raise


def store_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Store train and test datasets in the 'data/raw' directory inside the project."""
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_path = os.path.join(base_dir, "data")
        raw_data_dir = os.path.join(data_path, "raw")

        os.makedirs(raw_data_dir, exist_ok=True)  

        train_df.to_csv(os.path.join(raw_data_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(raw_data_dir, "test.csv"), index=False)

        LOGGER.debug("Train and test datasets saved at %s", raw_data_dir)
    except Exception as err:
        LOGGER.error("Unexpected error while saving data: %s", err)
        raise


def main():
    try:
        config = load_config(config_path="05_ml_pipeline_dvc_s3/params.yaml")
        test_ratio = config["data_ingestion"]["test_size"]

        dataset_url = "https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv"
        raw_data = fetch_data(csv_url=dataset_url)
        processed_data = clean_data(raw_data)

        train_df, test_df = train_test_split(processed_data, test_size=test_ratio, random_state=2)
        store_data(train_df, test_df)

    except Exception as err:
        LOGGER.error("Data ingestion process failed: %s", err)
        print(f"Error: {err}")


if __name__ == "__main__":
    main()