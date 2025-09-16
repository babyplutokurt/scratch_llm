# src/data_utils.py

import os
from datasets import load_dataset, concatenate_datasets
from itertools import chain

def prepare_pt_data(data_dir, tokenizer, save_path, num_proc=32, block_size=1024):
    """
    Prepares the pre-training dataset and saves it to disk.

    Args:
        data_dir (str): The directory where the datasets are stored.
        tokenizer: The tokenizer to use for tokenizing the text.
        save_path (str): The path to save the processed dataset.
        num_proc (int): The number of CPU cores to use.
        block_size (int): The block size for sequence packing.
    """
    pt_data_path = os.path.join(data_dir, "IndustryCorpus2")
    
    # Find all .parquet files in the directory
    data_files = []
    for root, dirs, files in os.walk(pt_data_path):
        for file in files:
            if file.endswith(".parquet"):
                data_files.append(os.path.join(root, file))

    if not data_files:
        raise ValueError(f"No .parquet files found in {pt_data_path}. Please run the data preparation script first.")

    dataset = load_dataset("parquet", data_files=data_files, split="train", columns=["text"])

    def preprocess_dataset(examples):
        """
        Preprocesses the pre-training dataset by appending the EOS token,
        tokenizing, concatenating, and chunking.
        """
        # Use the tokenizer's actual EOS token, not a hardcoded string
        eos_token = tokenizer.eos_token
        
        # Append EOS to each document
        text_examples = [text + eos_token for text in examples["text"] if text] # Ensure text is not empty
        
        # Tokenize without adding special tokens, as we've handled EOS manually
        tokenized_examples = tokenizer(text_examples, add_special_tokens=False)

        # Concatenate all tokenized examples into a single stream
        concatenated_examples = {
            k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()
        }
        
        # Determine the total number of tokens
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        
        # Round down to the nearest multiple of block_size
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        else:
            # Handle cases where the total text is smaller than a single block
            return {k: [] for k in concatenated_examples.keys()}

        # Split into chunks of block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        
        # For language modeling, the labels are the same as the inputs
        result["labels"] = result["input_ids"].copy()
        return result

    processed_dataset = dataset.map(
        preprocess_dataset,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )

    # Save the processed dataset to the specified path
    processed_dataset.save_to_disk(save_path)
    
    return processed_dataset

def prepare_sft_data(data_dir, tokenizer):
    """
    Prepares the supervised fine-tuning dataset.

    Args:
        data_dir (str): The directory where the datasets are stored.
        tokenizer: The tokenizer to use for tokenizing the text.

    Returns:
        A processed dataset ready for SFT.
    """
    sft_data_path = os.path.join(data_dir, "Infinity-Instruct")
    
    # Find all .parquet files in the directory
    data_files = []
    for root, dirs, files in os.walk(sft_data_path):
        for file in files:
            if file.endswith(".parquet"):
                data_files.append(os.path.join(root, file))

    if not data_files:
        raise ValueError(f"No .parquet files found in {sft_data_path}. Please run the data preparation script first.")

    dataset = load_dataset("parquet", data_files=data_files, split="train", columns=["conversations"])

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["conversations"])):
            messages = example["conversations"][i]
            # We will only use the first turn of the conversation for SFT.
            if len(messages) > 1 and messages[0]['from'] == 'human' and messages[1]['from'] == 'gpt':
                human_text = messages[0]['value']
                gpt_text = messages[1]['value']
                text = f"<|im_start|>user\n{human_text}<|im_end|>\n<|im_start|>assistant\n{gpt_text}<|im_end|>"
                output_texts.append(text)
        return output_texts

    return dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names, num_proc=32)


def prepare_dpo_data(data_dir, tokenizer):
    """
    Prepares the direct preference optimization dataset.

    Args:
        data_dir (str): The directory where the datasets are stored.
        tokenizer: The tokenizer to use for tokenizing the text.

    Returns:
        A processed dataset ready for DPO.
    """
    dpo_data_path = os.path.join(data_dir, "Infinity-Preference")
    
    # Find all .parquet files in the directory to include train and test splits
    data_files = []
    for root, dirs, files in os.walk(dpo_data_path):
        for file in files:
            if file.endswith(".parquet"):
                data_files.append(os.path.join(root, file))

    if not data_files:
        raise ValueError(f"No .parquet files found in {dpo_data_path}. Please run the data preparation script first.")

    dataset = load_dataset("parquet", data_files=data_files, split="train")

    def format_dpo_dataset(example):
        # Format the chosen and rejected responses
        prompt = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        chosen = f"{example['chosen'][1]['content']}<|im_end|>"
        rejected = f"{example['rejected'][1]['content']}<|im_end|>"
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    return dataset.map(format_dpo_dataset, remove_columns=dataset.column_names, num_proc=32)


if __name__ == '__main__':
    # This is for testing purposes.
    # You can run this script to see the data preparation process.
    from transformers import AutoTokenizer

    # We need a tokenizer to test the data preparation.
    # We will use the qwen tokenizer, as it's a good starting point.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    
    # Set pad token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # The data needs to be downloaded first.
    # You can run the scripts/prepare_data.sh script to download the data.
    # For this test, we will assume the data is in the ../data directory.
    data_dir = "../data"
    if not os.path.exists(data_dir):
        print("Data directory not found. Please run the scripts/prepare_data.sh script first.")
    else:
        print("Preparing PT data...")
        pt_dataset = prepare_pt_data(data_dir, tokenizer)
        print(pt_dataset)
        print(pt_dataset[0])

        print("\nPreparing SFT data...")
        sft_dataset = prepare_sft_data(data_dir, tokenizer)
        print(sft_dataset)
        print(sft_dataset[0])

        print("\nPreparing DPO data...")
        dpo_dataset = prepare_dpo_data(data_dir, tokenizer)
        print(dpo_dataset)
        print(dpo_dataset[0])
