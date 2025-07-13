import numpy as np
import pandas as pd
import logging
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def train_model(X_train: np.ndarray, y_train: np.ndarray, params : dict) -> RandomForestClassifier:
    """
    Trains a Random Forest model on the provided training data.
    
    Parameters:
    X_train (np.ndarray): Training features.
    y_train (np.ndarray): Training labels.
    params (dict): Dictionary containing model parameters such as 'n_estimator' and 'random_state'.
    
    Returns:
    RandomForestClassifier: The trained model.
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in X_train and y_train must match.")
        
        logger.info(f'initializing the model with params as {params}')

        model = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])

        logger.debug(f'model training started with {X_train.shape[0]} samples and {X_train.shape[1]} features')
        model.fit(X_train, y_train)
        logger.info('Model training completed successfully')

        return model
    except Exception as e:
        logger.error(f"Error in train_model function: {e}")
        raise e         

def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        # Load the training data
        train_data = pd.read_csv('./data/feature_engineered_data/train_tfidf.csv')
        X_train = train_data.drop(columns=['label']).values
        y_train = train_data['label'].values

        logger.info('Training data loaded successfully')
        logger.debug(f'Training data shape: {X_train.shape}, Labels shape: {y_train.shape}')

        # Define model parameters
        params = {
            'n_estimators': 40,
            'random_state': 2
        }

        # Train the model
        model = train_model(X_train, y_train, params)

        # Save the trained model
        save_model(model, './models/random_forest_model.pkl')

    except Exception as e:
        logger.error(f"An error occurred in main function: {e}")
        raise e
    

if __name__ == "__main__":
    main()




        

        

