
# Project Running Guide

This document provides instructions to set up and run the project code in both single-node and multi-node environments. Please follow the steps below carefully.

---

## 0. Environment Setup

Before running the project, ensure the following:

0. **Login to Hugging Face Hub**
   ```bash
   huggingface-cli login
   export WANDB_API_KEY=e2b4ba9bc9cf888b34d16cc8b0ad33dcbc033eb2
   ```
  
1. **Dataset Preparation**

Download datasets from `rel-bench`.
   ```bash
   python PretrainingDataset.py
   ```

2. **Data Preprocessing**


Precompute tokenized inputs for the Qwen model. Run the following commands **in parallel** (for different epochs):
   ```bash
   python pre_data_tokenizer.py --save_path ./data/precomputed/epoch1/ &
   python pre_data_tokenizer.py --save_path ./data/precomputed/epoch2/ &
   python pre_data_tokenizer.py --save_path ./data/precomputed/epoch3/ &
   python pre_data_tokenizer.py --save_path ./data/precomputed/epoch4/ &
   python pre_data_tokenizer.py --save_path ./data/precomputed/epoch5/ &
   ```

4. **Training on a Single Node**

To launch training on a single node with 8 processes:
```bash
accelerate launch \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port 0 \
  PreTrainer_ddp.py
```

If there are issues happened when downloading Qwen 0.5B model, then it may be caused by different cache dirs across different ranks. set env variables as:

```
export HF_HOME=/your/cache/path
export TRANSFORMERS_CACHE=$HF_HOME
```

This may resolve the issue.

4. Training on Multiple Nodes

For distributed training across multiple nodes (example: 4 nodes, 32 processes in total):

```bash
  accelerate launch \
  --num_processes 32 \
  --num_machines 4 \
  --machine_rank $node_rank$ \           # e.g., 0-3
  --main_process_ip 10.0.0.1 \
  --main_process_port 29500 \
  --mixed_precision bf16 \
  PreTrainer_multinode.py
```

