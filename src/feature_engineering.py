import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """load parameters from a json file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            return params
        logger.info(f"Parameters loaded successfully from {params_path}")
    except FileNotFoundError as e:
        logger.error(f"Parameters file not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise 
    
def tfidf(train_data : pd.DataFrame,test_data : pd.DataFrame, max_features :int) -> tuple:
    """
    applies tfidf vectorization on the train and test data"""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train = train_data['text'].values
        X_test = test_data['text'].values
        y_train = train_data['target'].values
        y_test = test_data['target'].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test

        logger.debug('tfidf applied and data transformed')
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error in tfidf function: {e}")
        raise 

def main():

    try:
        """loads the train and test data, applies tfidf vectorization and saves the transformed data"""
        train_data = pd.read_csv('./data/processed_data/train_processed.csv')
        test_data = pd.read_csv('./data/processed_data/test_processed.csv')
        
        params = load_params(params_path='params.yaml')

        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)

        logger.info('Train and test data loaded successfully')
        logger.debug(f'Train data shape: {train_data.shape}, Test data shape: {test_data.shape}')

        max_features = params['feature_engineering']['max_features']
        train_df, test_df = tfidf(train_data, test_data, max_features)

        data_path = os.path.join('data', 'feature_engineered_data')
        os.makedirs(data_path, exist_ok=True)

        train_df.to_csv(os.path.join(data_path, 'train_tfidf.csv'), index = False)
        test_df.to_csv(os.path.join(data_path, 'test_tfidf.csv'), index = False)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()










