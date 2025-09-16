# src/run_sft.py

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from data_utils import prepare_sft_data
from accelerate import Accelerator # Import the Accelerator

def main():
    # Initialize the Accelerator
    accelerator = Accelerator()

    parser = argparse.ArgumentParser(description="Supervised fine-tuning of a small LLM.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--data_dir", type=str, default="../data", help="Directory where the datasets are stored.")
    parser.add_argument("--output_dir", type=str, default="../models/sft", help="Directory to save the fine-tuned model.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of gradient accumulation steps.")
    parser.add_argument("--logging_steps", type=int, default=20, help="Log every X updates steps.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length.")
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

    # Prepare dataset using the main_process_first context manager
    with accelerator.main_process_first():
        sft_dataset = prepare_sft_data(args.data_dir, tokenizer)

    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=args.logging_steps,
        bf16=True,
        report_to="tensorboard",
        max_seq_length=args.max_seq_length,
        packing=True,
    )

    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=sft_dataset,
        args=training_args,
        dataset_text_field="text",
    )

    # Start training
    trainer.train()

    # Save the final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()