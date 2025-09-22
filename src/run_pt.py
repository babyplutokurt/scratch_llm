# src/run_pt.py

import os
import argparse
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM
import torch
from torch.distributed.elastic.multiprocessing.errors import record


@record
def main():
    # The Trainer will automatically handle the distributed environment and logging.
    parser = argparse.ArgumentParser(description="Pre-train a small LLM from scratch.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where the datasets are stored.")
    parser.add_argument("--output_dir", type=str, default="./models/pt", help="Directory to save the trained model.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=12, help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of gradient accumulation steps.")
    parser.add_argument("--logging_steps", type=int, default=20, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=10000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the local model and tokenizer.")
    args = parser.parse_args()

    # All processes will now load the tokenizer from the specified local path
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model from a modified config, similar to mini_qwen
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    # Modify the configuration to create a smaller model
    config.hidden_size = 512
    config.intermediate_size = 2048
    config.num_hidden_layers = 28
    config.num_attention_heads = 8
    config.num_key_value_heads = 4
    
    model = AutoModelForCausalLM.from_config(
        config,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    # The data is now expected to be pre-processed and saved to disk.
    processed_data_path = os.path.join(args.data_dir, "processed", "pt")
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(
            f"Processed data not found at {processed_data_path}. "
            "Please run the `src/preprocess_pt.py` script first."
        )
    
    pt_dataset = load_from_disk(processed_data_path)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        bf16=True,
        report_to="tensorboard",
        ddp_find_unused_parameters=False
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=pt_dataset,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

    # Save the final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()