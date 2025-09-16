# src/run_dpo.py

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from data_utils import prepare_dpo_data

def main():
    parser = argparse.ArgumentParser(description="Direct Preference Optimization of a small LLM.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the SFT model.")
    parser.add_argument("--data_dir", type=str, default="../data", help="Directory where the datasets are stored.")
    parser.add_argument("--output_dir", type=str, default="../models/dpo", help="Directory to save the DPO model.")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="Learning rate for training.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of gradient accumulation steps.")
    parser.add_argument("--logging_steps", type=int, default=20, help="Log every X updates steps.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum prompt length.")
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

    # Prepare dataset
    dpo_dataset = prepare_dpo_data(args.data_dir, tokenizer)

    # Training arguments
    training_args = DPOConfig(
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
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    # Initialize Trainer
    trainer = DPOTrainer(
        model=model,
        train_dataset=dpo_dataset,
        args=training_args,
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()

    # Save the final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
