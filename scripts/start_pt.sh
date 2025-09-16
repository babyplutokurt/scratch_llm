#!/bin/bash

# This script starts the pre-training phase of the scratch_llm project.

# Check if processed data exists, if not, run preprocessing first.
if [ ! -d "./data/processed/pt" ]; then
    echo "Processed data not found. Running preprocessing script first..."
    bash scripts/run_preprocess.sh
fi

# To switch between training strategies, change the --config_file argument to:
# --config_file accelerate_config_ddp.yaml  (Faster for smaller models, uses more VRAM)
# --config_file accelerate_config_fsdp.yaml (Slower for smaller models, saves VRAM for huge models)

# --- DDP Configuration (Default for this model size) ---
# Global Batch Size = 32 (per_device) * 8 (gpus) * 16 (accumulation) = 4096
accelerate launch --config_file accelerate_config_ddp.yaml src/run_pt.py \
    --output_dir ./models/pt \
    --data_dir ./data \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 16
