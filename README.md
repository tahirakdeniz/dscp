# Hybrid Vulnerability Analysis in AI Code Generation Using Software Metrics and LLM

**Authors:** Muhammed Tahir Akdeniz, İbrahim Enes Köse

---

## Abstract

*Abstract of Paper comes here. (Copy and paste your abstract.)*

---

## Overview

This repository implements a hybrid vulnerability analysis approach that combines software metrics extraction and LLM-based embedding generation to analyze AI code generation vulnerabilities. The workflow includes several key steps:

1. **Data Collection & Preprocessing**  
2. **Metric Extraction**  
3. **LLM Embedding Extraction**  
4. **Feature Combination**  
5. **XGBoost-based Model Training & Evaluation**  

Each stage is implemented as separate scripts or notebooks, allowing for a modular and manageable pipeline.

---

## Data Collection

- **Dataset Source:**  
    The analysis uses the AI v2 dataset provided by [FormAI-Dataset](https://github.com/FormAI-Dataset).

- **Process Overview:**  
    1. **Convert Classification JSON to CSV:**  
         Use the `classification_json_to_csv.ipynb` notebook to process the given classification JSON file and generate a CSV file containing the dataset vulnerabilities.
    
    2. **Clean the Dataset:**  
         Remove unknown or missing values, duplicate entries, and keep only the known values using the cleaning routines in `clean_classification_csv`.
    
    3. **Dataset Splitting:**  
         Generate grouped counts for a balanced dataset and split the data using the `explore_and_split_classification_csv.ipynb` notebook. The resulting CSV files are saved in the `data` folder.

---

## Metric Extraction

- **Script:** `software_metrics/metric.py`  
- **Purpose:**  
    Extract software metrics from dataset files, processing them to generate features for both training and testing.

- **How to Run:**  
    ```bash
    python software_metrics/metric.py <directory_path> <csv_train_file> <csv_test_file>
    ```
    - `<directory_path>`: Path to the dataset files (should point to FormAI-v2-DATASET/DATASETv2 in the FormAI dataset).
    - `<csv_train_file>` and `<csv_test_file>`: Paths to the CSV files for training and testing data.

---

## LLM Embedding Extraction

- **Script:** `llm/batch_llm_extract_embeeddings.py`  
- **Purpose:**  
    Extract embeddings from source code files using an LLM, which will be combined with software metrics for further analysis.

- **Key Arguments:**
    - `--csv_file`: Path to the CSV file that lists file names and metadata.
    - `--source_folder`: Directory containing the source code files.
    - `--output_folder`: Directory where the generated embeddings will be saved.

- **How to Run:**  
    ```bash
    python llm/batch_llm_extract_embeeddings.py --csv_file <csv_file> --source_folder <source_folder> --output_folder <output_folder>
    ```

---

## Combining Features

- **Script:** `combined_features/combine.py`  
- **Purpose:**  
    Combine the software metrics and LLM-generated embeddings into a unified feature set for model training.

- **Key Arguments:**
    - `--csv_path`: Path to the CSV file containing the code metrics.
    - `--embed_dir`: Directory where the embedding files are stored.
    - `--output_dir`: Directory to save the combined feature files.

- **How to Run:**  
    ```bash
    python combined_features/combine.py --csv_path <csv_path> --embed_dir <embed_dir> --output_dir <output_dir>
    ```

---

## XGBoost Modeling

- **Script:** `xgboost/model.py`  
- **Purpose:**  
    Load the combined features and train/test an XGBoost model for vulnerability analysis.

- **Key Arguments:**
    - `--combined_path`: Path to the directory containing .npy files with the combined features.
    - `--train_csv`: Path to the CSV file containing training data information.
    - `--test_csv`: Path to the CSV file containing testing data information.

- **How to Run:**  
    ```bash
    python xgboost/model.py --combined_path /path/to/npy/files --train_csv /path/to/train.csv --test_csv /path/to/test.csv
    ```

---

## Summary

This repository provides a complete pipeline for vulnerability analysis in AI code generation:

- **Data Collection & Cleaning:** Process and split the dataset for balanced analysis.  
- **Metric Extraction & Embedding Generation:** Extract software metrics and LLM embeddings.  
- **Feature Combination:** Merge the features from different sources.  
- **Model Training:** Apply XGBoost to analyze the combined features for vulnerability detection.

Feel free to explore each module and adjust the parameters as needed for your specific analysis.