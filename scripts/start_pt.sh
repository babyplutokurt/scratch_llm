#!/bin/bash

# Start whole pre-training pipeline.

# IMPORTANT: Ensure you have downloaded the model assets first. For example:
# python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='./models/Qwen3-0.6B', local_dir_use_symlinks=False)"

# Ensure processed dataset exists.
if [ ! -d "./data/processed/pt" ]; then
    echo "Processed data not found. Running preprocessing first..."
    bash scripts/run_preprocess.sh
fi

mkdir -p ./logs

export PYTHONUNBUFFERED=1

NCCL_NET=Socket \
NCCL_IB_DISABLE=1 \
NCCL_SHIMNET_DISABLE=1 \
NCCL_NET_PLUGIN=none \
NCCL_PLUGIN_DISABLE=1 \
accelerate launch \
    --config_file accelerate_config_ddp.yaml \
    --num_processes 8 \
    --main_process_port 29500 \
    src/run_pt.py \
        --model_path ./models/Qwen3-0.6B \
        --output_dir ./models/pt \
        --data_dir ./data \
        --per_device_train_batch_size 24 \
        --gradient_accumulation_steps 16