import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

nltk.download('stopwords')
nltk.download('punkt_tab')

# Ensure the log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok = True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text: str) -> str:
    """Transforms the input text by removing punctuation, converting to lowercase, 
    removing stopwords, and applying stemming."""
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english')]
    text = [ps.stem(word) for word in text]
    return ' '.join(text)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug('starting data preprocessing')
        encoder = LabelEncoder()
        df['target'] = encoder.fit_transform(df['target'])
        logger.info(f"Label encoding completed")

        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')

        df['text'] = df['text'].apply(transform_text)
        logger.info('Text transformation completed')
        logger.info(f'text processing example: {df["text"].iloc[0]}')
        return df
    except KeyError as e:
        logger.error(f"Column not found during preprocessing: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during preprocessing: {e}")
        raise

def main():
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        raw_test_data_path = './data/raw_data/test_data.csv'
        raw_train_data_path = './data/raw_data/train_data.csv'
        logger.debug('Data loaded properly')

        train_data = pd.read_csv(raw_train_data_path)
        test_data = pd.read_csv(raw_test_data_path)
        logger.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")

        train_processed_data = preprocess_data(train_data)
        test_processed_data = preprocess_data(test_data)

        logger.info(f"Processed train data shape: {train_processed_data.shape}, Processed test data shape: {test_processed_data.shape}")

        data_path = os.path.join("./data", 'processed_data')
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logger.info(f"Saving processed data to {data_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main()


















