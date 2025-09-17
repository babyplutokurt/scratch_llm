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
export TOKENIZERS_PARALLELISM=false
export ACCELERATE_LOG_LEVEL=info
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DISTRIBUTED_BACKEND=nccl

main_port=${MAIN_PROCESS_PORT:-29500}

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export PYTHONFAULTHANDLER=1
export HF_HUB_DISABLE_TELEMETRY=1

accelerate launch \
    --config_file accelerate_config_ddp.yaml \
    --num_processes 8 \
    --main_process_port "${main_port}" \
    src/run_pt.py \
        --model_path ./models/Qwen3-0.6B \
        --output_dir ./models/pt \
        --data_dir ./data \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 16
