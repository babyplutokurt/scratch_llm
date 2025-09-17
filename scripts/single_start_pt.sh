CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
PYTHONFAULTHANDLER=1 \
HF_HUB_DISABLE_TELEMETRY=1 \
python -u src/run_pt.py \
    --model_path ./models/Qwen3-0.6B \
    --output_dir ./models/pt \
    --data_dir ./data \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 10000