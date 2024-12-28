from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import logging
import argparse

def combine_features_single_file(code_metrics, embeddings):
    """
    Combines CodeBERT embeddings with code metrics features.
    
    Parameters:
    codebert_embedding: numpy array of shape (1, 768)
    code_metrics: numpy array of shape (36,)
    
    Returns:
    combined_features: numpy array of shape (1, 804) - concatenated features
    """
    # Reshape code metrics to match embedding dimension
    code_metrics_reshaped = code_metrics.reshape(1, -1)
    
    # Concatenate features along the second axis
    combined_features = np.concatenate([embeddings, code_metrics_reshaped], axis=1)
    
    return combined_features

def combine_features(metrics, embedding):
    """
    Combines code metrics with LLM embeddings for vulnerability detection.
    
    Parameters:
    metrics: np.array - Array of code metrics (37 features)
    embedding: np.array - LLM generated embedding vector
    
    Returns:
    np.array - Combined feature vector ready for XGBoost
    """
    return np.array([
        combine_features_single_file(code_metrics, llm_embedding)
        for code_metrics, llm_embedding in zip(metrics, embedding)
    ])

# def train_vulnerability_detector(combined_features,y_labels):
#     """
#     Trains XGBoost model with combined features.
    
#     Parameters:
#     X_metrics: np.array - Array of code metrics for all samples
#     X_embeddings: np.array - Array of LLM embeddings for all samples
#     y_labels: np.array - Binary labels (1 for vulnerable, 0 for safe)
    
#     Returns:
#     xgb.XGBClassifier - Trained model
#     """
    
#     # Initialize and train XGBoost
#     model = xgb.XGBClassifier(
#         n_estimators=100,
#         max_depth=6,
#         learning_rate=0.1,
#         objective='binary:logistic',
#         eval_metric='auc'
#     )
    
#     model.fit(combined_features, y_labels)
#     return model

def combine_features_for_files(csv_path, embed_dir,output_dir):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Read the CSV file
    logging.info(f"Reading CSV file from {csv_path}")
    code_metrics_df = pd.read_csv(csv_path)
    
    combined_features_list = []
    
    for index, row in code_metrics_df.iterrows():
        file_name = row['file_name']  # Assuming there's a column 'file_name' in the CSV
        metrics = row.iloc[2:38].astype(float).to_numpy()  # Adjust the column indices as needed
        
        # Load the corresponding embedding
        embed_path = f"{embed_dir}/{file_name}_embedding.npy"
        if not os.path.exists(embed_path):
            logging.warning(f"Embedding file not found for {file_name}")
            continue
        embeddings = np.load(embed_path)
        
        # Combine the metrics and embeddings
        combined_features = combine_features_single_file(metrics, embeddings)
        
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the combined features as a .npy file
        combined_file_path = f"{output_dir}/{file_name}_combined.npy"
        np.save(combined_file_path, combined_features)
        
        logging.debug(f"Combined features saved to {combined_file_path}")
        
        combined_features_list.append(combined_features)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Combine code metrics with embeddings.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file containing code metrics.')
    parser.add_argument('--embed_dir', type=str, required=True, help='Directory containing the embedding files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the combined feature files.')
    return parser.parse_args()

if "__main__" == __name__:    

    # Parse arguments
    args = parse_arguments()
    csv_path = args.csv_path
    embed_dir = args.embed_dir
    output_dir = args.output_dir
    
    combine_features_for_files(csv_path, embed_dir,output_dir)
