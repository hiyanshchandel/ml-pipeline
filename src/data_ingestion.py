import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging

#ensure the log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok = True)

#logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url : str) -> pd.DataFrame:
    """load data from a given url"""
    try:
        logger.debug(f"loading data from {data_url}")
        df = pd.read_csv(data_url)
        logger.info(f"data loaded successfully with shape {df.shape}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("No data found in file.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    try:
        df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace =True, axis = 1)
        df.rename(columns = {'v1' : 'target', 'v2' : 'text'}, inplace = True)
        logger.info(f"Data preprocessing completed successfully with shape {df.shape}.")
        return df
    except KeyError as e: #when a column is not found
        logger.error(f"Column not found during preprocessing: {e}")
        raise
    except Exception as e: #for any other unexpected error
        logger.error(f"An unexpected error occurred during preprocessing: {e}")
        raise

def save_data(train_data : pd.DataFrame, test_data : pd.DataFrame, data_path: str) -> None:
    """save the dataframe to a csv file"""
    try:
        raw_data_path = os.path.join(data_path, 'raw_data')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train_data.csv'), index = False)
        test_data.to_csv(os.path.join(raw_data_path, 'test_data.csv'), index = False)
        logger.info(f"Data saved successfully at {raw_data_path}")

    except Exception as e:
        logger.error(f"An error occurred while saving data: {e}")
        raise

def main():
    try:
        test_size = 0.20
        data_url = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
        df = load_data(data_url)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')
        logger.info("Data ingestion process completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the data ingestion process: {e}")
        raise

if __name__ == "__main__":
    main()