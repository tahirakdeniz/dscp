import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import numpy as np
import warnings
import sys
import logging
import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING messages

# Suppress specific PyTorch UserWarning about Flash Attention
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

current_date = datetime.datetime.now().strftime("%Y-%m-%d")
logfilename = f"extract_embeddings_{current_date}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=logfilename)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate embeddings using CodeBERT and CodeT5.")
    parser.add_argument('file_path', type=str, help='Path to the source code file.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save embeddings. Defaults to the input file\'s directory.')
    return parser.parse_args()

def load_model(model_name, model_type, device):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_type == 'AutoModel':
            model = AutoModel.from_pretrained(model_name).to(device)
        elif model_type == 'AutoModelForSeq2SeqLM':
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        model.eval()
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return None, None

def split_code(code, tokenizer, max_length=512, stride=50):
    """
    Splits the code into chunks that fit the model's maximum sequence length.
    """
    # Encode the entire code with overflow handling
    encoding = tokenizer(
        code,
        return_overflowing_tokens=True,
        max_length=max_length,
        stride=stride,
        truncation=True,
        return_tensors='pt',
        padding='max_length'
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    num_chunks = input_ids.shape[0]
    
    return input_ids, attention_mask, num_chunks

def generate_embedding(tokenizer, model, code, device, model_type):
    try:
        with torch.no_grad():
            # Split code into manageable chunks
            input_ids, attention_mask, num_chunks = split_code(code, tokenizer)
            all_embeddings = []

            for i in range(num_chunks):
                # Select the i-th chunk
                chunk_input_ids = input_ids[i].unsqueeze(0).to(device)
                chunk_attention_mask = attention_mask[i].unsqueeze(0).to(device)
                
                if model_type == 'AutoModelForSeq2SeqLM':
                    # For seq2seq models like CodeT5, get encoder outputs
                    encoder_outputs = model.encoder(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask)
                    embedding = encoder_outputs.last_hidden_state[:, 0, :].cpu().numpy()
                else:
                    # For models like CodeBERT
                    outputs = model(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask)
                    if hasattr(outputs, 'last_hidden_state'):
                        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    elif hasattr(outputs, 'pooler_output'):
                        embedding = outputs.pooler_output.cpu().numpy()
                    else:
                        raise ValueError("Model output does not have last_hidden_state or pooler_output.")
                all_embeddings.append(embedding)
            
            # Aggregate embeddings (e.g., average)
            aggregated_embedding = np.mean(all_embeddings, axis=0)
            return aggregated_embedding
    except RuntimeError as e:
        if 'out of memory' in str(e):
            logger.error("CUDA out of memory. Switching to CPU.")
            torch.cuda.empty_cache()
            return None
        else:
            raise e
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        return None

def process_file(file_path, output_dir):
    # Define subfolders for each model
    models = {
        'CodeBERT': {'name': 'microsoft/codebert-base', 'type': 'AutoModel'},
        'CodeT5': {'name': 'Salesforce/codet5-base', 'type': 'AutoModelForSeq2SeqLM'}
    }

    # Create subfolders
    for model_label in models.keys():
        model_output_dir = os.path.join(output_dir, model_label)
        os.makedirs(model_output_dir, exist_ok=True)

    # Read the source code
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return  # Changed from sys.exit(1) to continue processing other files

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Initial device: {device}")

    for model_label, model_info in models.items():
        model_name = model_info['name']
        model_type = model_info['type']
        logger.info(f"\nProcessing with {model_label} ({model_name})")

        current_device = device
        tokenizer, model = load_model(model_name, model_type, current_device)

        if tokenizer is None or model is None:
            logger.error(f"Skipping {model_label} due to loading issues.")
            continue

        embedding = generate_embedding(tokenizer, model, code, current_device, model_type)

        # If embedding is None, it means OOM occurred on GPU. Retry on CPU.
        if embedding is None and device == 'cuda':
            current_device = 'cpu'
            logger.info(f"Retrying {model_label} on CPU.")
            tokenizer, model = load_model(model_name, model_type, current_device)
            if tokenizer is None or model is None:
                logger.error(f"Skipping {model_label} due to loading issues on CPU.")
                continue
            embedding = generate_embedding(tokenizer, model, code, current_device, model_type)
            if embedding is None:
                logger.error(f"Failed to generate embedding for {model_label} on CPU.")
                continue

        # Define the embedding file path
        model_output_dir = os.path.join(output_dir, model_label)
        base_filename = os.path.basename(file_path)
        embedding_file = os.path.join(model_output_dir, f"{base_filename}_embedding.npy")

        # Save the embedding
        try:
            np.save(embedding_file, embedding)
            logger.info(f"Saved {model_label} embedding to {embedding_file}")
            logger.info(f"Embedding shape: {embedding.shape}")
        except Exception as e:
            logger.error(f"Error saving embedding for {model_label}: {e}")

        # Free up GPU memory if used
        if current_device == 'cuda':
            del tokenizer
            del model
            torch.cuda.empty_cache()

def main():
    args = parse_arguments()
    file_path = args.file_path

    if not os.path.isfile(file_path):
        logger.error(f"File {file_path} does not exist.")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(os.path.abspath(file_path))

    process_file(file_path, output_dir)

if __name__ == "__main__":
    main()