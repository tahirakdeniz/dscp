import numpy as np
import pandas as pd
import os
import logging
import argparse
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(npy_path, train_csv, test_csv):
    # Read the CSV files
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    # Initialize lists to hold data and labels
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    
    # Load training data
    for index, row in train_df.iterrows():
        try:
            file_path = os.path.join(npy_path, row['file_name']+"_embedding.npy")
            data = np.load(file_path)
            train_data.append(data)
            train_labels.append(row['vulnerability'])
        except Exception as e:
            logger.error(f"Error loading {row['file_name']}")
        
    # Load testing data
    for index, row in test_df.iterrows():
        try:
            file_path = os.path.join(npy_path, row['file_name']+"_embedding.npy")
            data = np.load(file_path)
            test_data.append(data)
            test_labels.append(row['vulnerability'])
        except:
            logger.error(f"Error loading {row['file_name']}")
    
    # Convert lists to numpy arrays
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    return (train_data, train_labels), (test_data, test_labels)

def train_xgboost(train_data, train_labels, test_data, test_labels):
    
        # Ensure train_data is a 2-dimensional array
    if len(train_data.shape) != 2:
        train_data = np.reshape(train_data, (train_data.shape[0], -1))
        test_data = np.reshape(test_data, (test_data.shape[0], -1))
    
    # Ensure train_labels is a 1-dimensional array
    if len(train_labels.shape) != 1:
        train_labels = np.reshape(train_labels, (train_labels.shape[0],))
        test_labels = np.reshape(test_labels, (test_labels.shape[0],))
    
    # Initialize the model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Train the model
    model.fit(train_data, train_labels)
    

    # Make predictions
    y_pred = model.predict(test_data)
    
        
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(test_labels, y_pred),
        'precision': precision_score(test_labels, y_pred),
        'recall': recall_score(test_labels, y_pred),
        'f1': f1_score(test_labels, y_pred)
    }
    
    logger.info(f"Metrics: {metrics}")

    
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Load dataset and print shapes of training and testing data.")
    parser.add_argument('--combined_path', type=str, required=True, help='Path to the directory containing .npy files')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to the training CSV file')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to the testing CSV file')
    return parser.parse_args()



if "__main__" == __name__:
    
    args = parse_arguments()
    
    combined_path = args.combined_path
    train_csv = args.train_csv
    test_csv = args.test_csv
    
    (train_data, train_labels), (test_data, test_labels) = load_dataset(combined_path, train_csv, test_csv)
    
    train_xgboost(train_data, train_labels, test_data, test_labels)
    
    
    
    