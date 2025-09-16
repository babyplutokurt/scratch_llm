# src/preprocess_pt.py

import os
import argparse
from transformers import AutoTokenizer
from data_utils import prepare_pt_data

def main():
    """
    This script is dedicated to running the pre-training data preprocessing step.
    It loads the raw data, processes it using the prepare_pt_data function,
    and saves the final dataset to disk for the training script to use.
    """
    parser = argparse.ArgumentParser(description="Pre-process the pre-training dataset.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where the raw datasets are stored.")
    parser.add_argument("--save_path", type=str, default="./data/processed/pt", help="Directory to save the processed dataset.")
    parser.add_argument("--num_proc", type=int, default=32, help="Number of CPU cores to use for processing.")
    args = parser.parse_args()

    print(f"Starting pre-training data processing with {args.num_proc} cores...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run the data preparation and save the dataset
    prepare_pt_data(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        save_path=args.save_path,
        num_proc=args.num_proc
    )

    print(f"Successfully processed and saved the dataset to {args.save_path}")

if __name__ == "__main__":
    main()
