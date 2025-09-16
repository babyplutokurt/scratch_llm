#!/bin/bash

# This script starts the direct preference optimization phase of the scratch_llm project.

# Make sure to run this script from the root of the scratch_llm project.
# Example: ./scripts/start_dpo.sh

accelerate launch --config_file accelerate_config.yaml src/run_dpo.py \
    --model_path ./models/sft \
    --output_dir ./models/dpo \
    --data_dir ./data
