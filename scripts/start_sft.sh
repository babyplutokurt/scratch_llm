#!/bin/bash

# This script starts the supervised fine-tuning phase of the scratch_llm project.

# Make sure to run this script from the root of the scratch_llm project,
# and only after the pre-training phase is complete.

# --- Environment Variable for Optimal Data Processing ---
export TOKENIZERS_PARALLELISM=false

# --- DDP Configuration (Optimal for this model size) ---
# Global Batch Size = 16 (per_device) * 8 (gpus) * 16 (accumulation) = 2048
accelerate launch --config_file accelerate_config_ddp.yaml src/run_sft.py \
    --model_path ./models/pt \
    --output_dir ./models/sft \
    --data_dir ./data \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16