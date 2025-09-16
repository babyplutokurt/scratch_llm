#!/bin/bash

# This script runs the CPU-intensive data preprocessing for all datasets.
# Run this script ONCE before starting any training jobs.

# --- Environment Variable for Optimal Data Processing ---
# Disables the tokenizer's internal parallelism to let `datasets.map` manage CPU resources.
export TOKENIZERS_PARALLELISM=false

echo "--- Starting Pre-training Data Preprocessing ---"
python src/preprocess_pt.py --num_proc 32

# echo "--- Starting SFT Data Preprocessing (Placeholder for future) ---"
# python src/preprocess_sft.py --num_proc 32

# echo "--- Starting DPO Data Preprocessing (Placeholder for future) ---"
# python src/preprocess_dpo.py --num_proc 32

echo "--- All data preprocessing complete. ---"
