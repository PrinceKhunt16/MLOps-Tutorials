import os
import logging
import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Define project base directory (assuming script is inside src/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")

# Ensure the "logs" directory exists
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Logging configuration
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(LOG_DIR, "feature_engineering.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)
        logger.debug("Data loaded and NaNs filled from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise


def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data["text"].values
        y_train = train_data["target"].values
        X_test = test_data["text"].values
        y_test = test_data["target"].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df["label"] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df["label"] = y_test

        logger.debug("TF-IDF applied and data transformed successfully")
        return train_df, test_df
    except Exception as e:
        logger.error("Error during TF-IDF transformation: %s", e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug("Data saved to %s", file_path)
    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise


def main():
    try:
        params = load_params(PARAMS_PATH)
        max_features = params["feature_engineering"]["max_features"]

        train_data = load_data(os.path.join(INTERIM_DIR, "train_processed.csv"))
        test_data = load_data(os.path.join(INTERIM_DIR, "test_processed.csv"))

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, os.path.join(PROCESSED_DIR, "train_tfidf.csv"))
        save_data(test_df, os.path.join(PROCESSED_DIR, "test_tfidf.csv"))

        logger.info("Feature engineering process completed successfully")
    except Exception as e:
        logger.error("Failed to complete the feature engineering process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
