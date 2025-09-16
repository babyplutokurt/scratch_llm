# scratch_llm: Train your llm from scratch

This project aims to train a small Large Language Model (LLM) from scratch, following the principles and techniques demonstrated in the `mini_qwen` project. The training process is divided into three main stages:

1.  **Pre-training (PT):** Training the model on a large text corpus to learn general language understanding.
2.  **Supervised Fine-Tuning (SFT):** Fine-tuning the model on an instruction-based dataset to follow user commands.
3.  **Direct Preference Optimization (DPO):** Aligning the model with human preferences using a preference dataset.

This project is optimized for a multi-GPU environment, specifically for 8x H100 80GB GPUs.

## Setup

1.  **Create and activate a conda environment:**
    ```bash
    conda create -n scratch_llm python=3.10
    conda activate scratch_llm
    ```

2.  **Install dependencies in order:**
    To ensure packages with build-time dependencies like `flash-attn` install correctly, we need to install `torch` and `ninja` first.
    ```bash
    # Install PyTorch first, as it's a dependency for building other packages.
    pip install torch
    
    # Install ninja for faster build times.
    pip install ninja

    # Install the rest of the requirements.
    pip install -r requirements.txt
    ```

3.  **Log in to Hugging Face Hub:**
    The data download script uses `huggingface-cli` to download the datasets. You will need to log in with a Hugging Face account that has access to the required datasets.
    ```bash
    huggingface-cli login
    ```
