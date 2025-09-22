#!/bin/bash

# This script downloads all the datasets required for the scratch_llm project
# using the efficient huggingface-cli tool, with a robust loop to ensure all
# specified subdirectories are downloaded correctly.

# Make sure you have huggingface-cli installed and are logged in:
# pip install huggingface_hub
# huggingface-cli login

set -e # Exit immediately if a command exits with a non-zero status.

DATA_DIR="data"
echo "Creating data directory at $DATA_DIR"
mkdir -p $DATA_DIR

# --- 1. Pre-training data: IndustryCorpus2 (approx. 141GB) ---
echo "Downloading all specified 'high' quality subsets of BAAI/IndustryCorpus2..."
PT_REPO="BAAI/IndustryCorpus2"
PT_TARGET_DIR="$DATA_DIR/IndustryCorpus2"

declare -a all_pt_dirs=(
    "mathematics_statistics/chinese/high"
    "artificial_intelligence_machine_learning/chinese/high"
    "computer_programming_code/chinese/high"
    "news_media/chinese/high"
    "accommodation_catering_hotel/chinese/high"
    "computer_programming_code/english/high"
    "accommodation_catering_hotel/english/high"
    "artificial_intelligence_machine_learning/english/high"
    "tourism_geography/chinese/high"
    "film_entertainment/chinese/high"
    "news_media/english/high"
    "tourism_geography/english/high"
    "literature_emotion/chinese/high"
    "computer_communication/chinese/high"
    "current_affairs_government_administration/chinese/high"
    "film_entertainment/english/high"
    "literature_emotion/english/high"
    "computer_communication/english/high"
    "current_affairs_government_administration/english/high"
    "mathematics_statistics/english/high"
)

# Loop and download each directory individually to avoid issues with multiple --include flags.
for dir in "${all_pt_dirs[@]}"; do
    echo "Downloading files from $dir..."
    hf download "$PT_REPO" --repo-type dataset --local-dir "$PT_TARGET_DIR" \
        --include "$dir/*.parquet"
done

# --- 2. Supervised Fine-Tuning data: Infinity-Instruct ---
echo "Downloading the Infinity-Instruct dataset..."
hf download BAAI/Infinity-Instruct --repo-type dataset --local-dir "$DATA_DIR/Infinity-Instruct"

# --- 3. Direct Preference Optimization data: Infinity-Preference ---
echo "Downloading the Infinity-Preference dataset..."
hf download BAAI/Infinity-Preference --repo-type dataset --local-dir "$DATA_DIR/Infinity-Preference"

echo "All datasets have been downloaded into the $DATA_DIR directory."