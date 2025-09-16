# src/model.py

import torch
from transformers import AutoConfig, AutoModelForCausalLM

def create_model():
    """
    Creates a new, randomly initialized Qwen3-style model from a modified configuration.

    This function follows the methodology of the mini_qwen project:
    1. Load the configuration of a base model (Qwen3-0.6B).
    2. Modify the configuration to create a smaller, custom model.
    3. Instantiate a new model with random weights from this modified config.
    """
    # 1. Load the base configuration from the specified Qwen3 model
    base_model_name = "Qwen/Qwen3-0.6B"
    config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)

    # 2. Modify the configuration to create a smaller model as requested
    config.hidden_size = 512
    config.intermediate_size = 2048
    config.num_hidden_layers = 28   # Keeping the original depth
    config.num_attention_heads = 8 # Scaled down to maintain head dimension
    config.num_key_value_heads = 4 # Scaled down to maintain GQA ratio

    # The vocab size, rope_theta, etc., are inherited from the base model.
    
    print("Creating a new model from the following modified configuration:")
    print(config)

    # 3. Create a new model with random weights from this configuration
    model = AutoModelForCausalLM.from_config(
        config,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    
    return model

if __name__ == '__main__':
    # This is for testing purposes.
    # You can run this script to see the model architecture and parameter count.
    model = create_model()
    print("\nModel created successfully!")
    
    # Calculate and print the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal number of parameters: {num_params / 1e6:.2f}M")
    print(f"Trainable parameters: {num_trainable_params / 1e6:.2f}M")

    # Verify that the model is on bfloat16
    print(f"Model dtype: {next(model.parameters()).dtype}")
