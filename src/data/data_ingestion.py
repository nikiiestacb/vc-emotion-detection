import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml
from typing import Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def load_params() -> float:
    """Load configuration parameters from a YAML file.
    
    Returns:
        float: The test size parameter.
    """
    try:
        with open("C:/desktop/mlops/emotion_detection/params.yaml", "r") as f:  # Adjust the file path
            config = yaml.safe_load(f)
        return config["data_ingestion"]["test_size"]
    except FileNotFoundError:
        logger.error("The 'params.yaml' file was not found.")
        raise
    except KeyError:
        logger.error("The required key 'data_ingestion' or 'test_size' is missing in 'params.yaml'.")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the parameters: {e}")
        raise

def read_data(file_path: str) -> pd.DataFrame:
    """Read the CSV file into a DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        logger.info(f"🔹 Reading CSV from {file_path}...")
        df = pd.read_csv(file_path)
        logger.info("✅ CSV loaded.")
        return df
    except FileNotFoundError:
        logger.error(f"The file at {file_path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"The file at {file_path} is empty.")
        raise
    except Exception as e:
        logger.error(f"An error occurred while reading the CSV file: {e}")
        raise

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and preprocess the data.
    
    Args:
        df (pd.DataFrame): The original DataFrame.
        
    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    try:
        df.drop(columns=['tweet_id'], inplace=True)

        # Avoid SettingWithCopyWarning — use .copy()
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()

        # Replace values correctly
        final_df['sentiment'] = final_df['sentiment'].map({'happiness': 1, 'sadness': 0})

        logger.info("✅ Filtered and mapped sentiments.")
        return final_df
    except KeyError as e:
        logger.error(f"Missing expected column: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while processing the data: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save train and test data to the specified directory.
    
    Args:
        train_data (pd.DataFrame): The training data.
        test_data (pd.DataFrame): The testing data.
        data_path (str): The directory to save the files.
    """
    try:
        # Create directory path if it does not exist
        os.makedirs(data_path, exist_ok=True)
        
        # File paths
        train_path = os.path.join(data_path, "train.csv")
        test_path = os.path.join(data_path, "test.csv")

        # Save CSVs
        logger.info(f"💾 Saving train data to: {train_path}")
        train_data.to_csv(train_path, index=False)

        logger.info(f"💾 Saving test data to: {test_path}")
        test_data.to_csv(test_path, index=False)

        logger.info("✅ Data ingestion completed.")
    except Exception as e:
        logger.error(f"An error occurred while saving the data: {e}")
        raise

def main() -> None:
    """Main function to execute the data ingestion pipeline."""
    try:
        # Load parameters
        test_size = load_params()

        # Read data
        df = read_data(r"C:\Users\NikitaCB\Downloads\tweet_data.csv")  # Adjust the file path

        # Process data
        final_df = process_data(df)

        # Split the dataset
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        # Define the data path
        data_path = os.path.join("data", "raw")

        # Save processed data
        save_data(train_data, test_data, data_path)

    except Exception as e:
        logger.error(f"An error occurred during the pipeline execution: {e}")
        raise

if __name__ == "__main__":
    main()
