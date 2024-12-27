import argparse
import csv
import os
import sys
from tqdm import tqdm
from llm_extract_embeddings import process_file

def parse_arguments():
    parser = argparse.ArgumentParser(description="Batch generate embeddings using CodeBERT and CodeT5 based on a CSV file.")
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing file names and metadata.')
    parser.add_argument('--source_folder', type=str, required=True, help='Directory containing the source code files.')
    parser.add_argument('--output_folder', type=str, required=True, help='Directory to save the generated embeddings.')
    return parser.parse_args()

def read_csv(csv_file_path):
    """
    Reads the CSV file and returns a list of dictionaries representing each row.
    """
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
        return rows
    except FileNotFoundError:
        print(f"CSV file {csv_file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file {csv_file_path}: {e}")
        sys.exit(1)

def main():
    args = parse_arguments()
    csv_file = args.csv_file
    source_folder = args.source_folder
    output_folder = args.output_folder

    # Verify that source_folder exists
    if not os.path.isdir(source_folder):
        print(f"Source folder {source_folder} does not exist.")
        sys.exit(1)

    # Create output_folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the CSV file
    rows = read_csv(csv_file)
    total_files = len(rows)
    print(f"Total files to process: {total_files}")

    # Process each file listed in the CSV
    for row in tqdm(rows, desc="Processing files", unit="file"):
        file_name = row.get('file_name')
        if not file_name:
            print("Skipping a row with missing 'file_name'.")
            continue

        # Construct the full path to the source file
        source_file_path = os.path.join(source_folder, file_name)

        # Check if the source file exists
        if not os.path.isfile(source_file_path):
            print(f"Source file {source_file_path} does not exist. Skipping.")
            continue

        # Call the process_file function
        try:
            process_file(source_file_path, output_folder)
        except Exception as e:
            print(f"Error processing file {source_file_path}: {e}")
            continue

    print("Batch processing completed.")

if __name__ == "__main__":
    main()
