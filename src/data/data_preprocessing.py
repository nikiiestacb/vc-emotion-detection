import os
import re
import logging
import pandas as pd
import numpy as np
import nltk
import yaml
from typing import Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------- Set Up Logging ---------------------
log_dir = "logs"
log_file = os.path.join(log_dir, "text_preprocessing.log")

# Ensure the logs directory exists
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# --------------------- Load YAML Parameters ---------------------

def load_params(file_path: str) -> dict:
    """Loads parameters from a YAML file."""
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info("✅ Configuration loaded successfully.")
        return config
    except Exception as e:
        logging.error(f"❌ Error loading configuration: {e}")
        raise e  # Re-raise the error after logging

# --------------------- Download NLTK Resources ---------------------

try:
    nltk.download('wordnet')
    nltk.download('stopwords')
except Exception as e:
    logging.error(f"Failed to download NLTK resources: {e}")

# --------------------- Preprocessing Functions ---------------------

def lemmatization(text: str) -> str:
    """Lemmatizes words in the given text."""
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text: str) -> str:
    """Removes stop words from text."""
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text: str) -> str:
    """Removes numeric characters from text."""
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text: str) -> str:
    """Converts all text to lowercase."""
    return text.lower()

def removing_punctuations(text: str) -> str:
    """Removes punctuations and normalizes spaces."""
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "")
    return re.sub(r'\s+', ' ', text).strip()

def removing_urls(text: str) -> str:
    """Removes URLs from text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Applies full preprocessing pipeline on 'content' column."""
    try:
        df['content'] = df['content'].astype(str)  # Ensure string type
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logging.info("✅ Text normalization completed.")
    except Exception as e:
        logging.error(f"❌ Error in normalize_text: {e}")
    return df

# --------------------- Main Execution Block ---------------------

def main() -> None:
    # Load configuration parameters from YAML file
    try:
        config = load_params("C:/desktop/mlops/emotion_detection/params.yaml")
        data_path = config['data_paths']['raw_data_path']
        processed_data_path = config['data_paths']['processed_data_path']
    except Exception as e:
        logging.error(f"❌ Could not load configuration: {e}")
        return

    try:
        logging.info("🔹 Loading datasets...")
        train_data = pd.read_csv(data_path + 'train.csv')
        test_data = pd.read_csv(data_path + 'test.csv')
        logging.info("✅ Datasets loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"❌ File not found: {e}")
        return
    except Exception as e:
        logging.error(f"❌ Error reading CSVs: {e}")
        return

    # Process text
    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)

    # Save processed data
    try:
        os.makedirs(processed_data_path, exist_ok=True)
        train_processed_data.to_csv(os.path.join(processed_data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(processed_data_path, "test_processed.csv"), index=False)
        logging.info("💾 Processed data saved successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to save processed files: {e}")

if __name__ == "__main__":
    main()
