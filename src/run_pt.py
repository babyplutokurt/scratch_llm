# src/run_pt.py

import os
import argparse
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from accelerate import Accelerator # Import the Accelerator
from datasets import load_from_disk # Import load_from_disk
from model import create_model

def main():
    # Initialize the Accelerator
    accelerator = Accelerator()

    parser = argparse.ArgumentParser(description="Pre-train a small LLM from scratch.")
    parser.add_argument("--data_dir", type=str, default="../data", help="Directory where the datasets are stored.")
    parser.add_argument("--output_dir", type=str, default="../models/pt", help="Directory to save the trained model.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=12, help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of gradient accumulation steps.")
    parser.add_argument("--logging_steps", type=int, default=20, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=10000, help="Save checkpoint every X updates steps.")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model
    model = create_model()

    # The data is now expected to be pre-processed and saved to disk.
    # We load it directly from the specified path.
    processed_data_path = os.path.join(args.data_dir, "processed", "pt")
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(
            f"Processed data not found at {processed_data_path}. "
            "Please run the `src/preprocess_pt.py` script first."
        )
    
    print(f"Loading pre-processed dataset from {processed_data_path}...")
    pt_dataset = load_from_disk(processed_data_path)
    print("Dataset loaded successfully.")

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