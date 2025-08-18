import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from PretrainingDataset import RelBenchRowDataset  # Assuming you have this class or module
from typing import List
import json
from transformers import AutoTokenizer
import re
import torch.nn.functional as F
import argparse
import sys
# Initialize the accelerator


def collate_to_list(batch: List[str]) -> List[str]:
    return batch


def split_top_level(s: str) -> list[str]:
    parts = []
    buf = []
    level = 0
    in_quote = False
    quote_char = ""
    for c in s:
        if in_quote:
            buf.append(c)
            if c == quote_char:
                in_quote = False
        else:
            if c in ('"', "'"):
                in_quote = True
                quote_char = c
                buf.append(c)
            elif c in "[(":
                level += 1
                buf.append(c)
            elif c in "])":
                level -= 1
                buf.append(c)
            elif c == "," and level == 0:
                parts.append("".join(buf).strip())
                buf = []
            else:
                buf.append(c)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def get_prefix_text(s: str) -> str:
    """
    Split on top‑level commas, keep Dataset & Table always,
    then:
      - if only 0 or 1 further fields exist → return "Dataset…, Table…,"
      - otherwise split the rest in half and include the first half.
    """
    fields = split_top_level(s)
    fixed  = fields[:2]       # ["Dataset:…", "Table:…"]
    rest   = fields[2:]
    
    # Edge‐case: too few fields after Table
    if len(rest) <= 1:
        # join fixed with a trailing comma
        return ", ".join(fixed) + ","
    
    # Normal case: take half of the remaining fields
    half   = len(rest) // 2
    prefix = fixed + rest[:half]
    return ", ".join(prefix)


# Assuming you have a processor to handle the tokenization and dataset creation
class NumericDataProcessor:
    r"""
    Converts raw strings with angle‑bracket numbers into the tensors
    needed by the Numeric LLaMA model.

    • Any numeric literal wrapped in “< … >” is replaced by literal token
      “<NUM>”  (tokenised as a single ID).
    • Side‑arrays keep the float values and masks.
    """

    def __init__(self, tokenizer, max_length=256):
        self.tok = tokenizer
        self.max_len = max_length
        # convenience: fetch <NUM> id once
        self.num_id = self.tok("<NUM>")["input_ids"][0]
        self.NUM_RE = re.compile(
            r"<([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)>"
        )


    def make_dataset(
        self,
        texts: List[str],
        save_path: str,
        chunk_size: int = 50_000
    ) -> None:
        """
        Tokenize input texts and save the cumulative dataset every `chunk_size` samples.

        Args:
            texts (List[str]):
                List of raw text strings to be tokenized.
            save_path (str):
                Path where the torch.Tensor of all processed samples
                (up to the current point) will be saved. This file
                is overwritten at each save.
            chunk_size (int):
                Number of samples between successive saves.

        Returns:
            None. Side effect: writes a Tensor to `save_path` at every
            multiple of `chunk_size`, containing all samples processed so far.
        """
        
        accumulated: List[torch.Tensor] = []
        cnt = 1  # Counter to track the chunk files

        # Ensure the save directory exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for idx, text in enumerate(tqdm(texts, desc="Tokenizing texts"), start=1):
            encoded = self._encode_one(text)  # -> torch.Tensor of one example
            accumulated.append(encoded)

            # On every chunk boundary (or at the very end), save all so far:
            if idx % chunk_size == 0 and idx > 10:
                # Save the accumulated data in a new file with chunk count
                chunk_file_path = os.path.join(save_path, f"processed_data_{cnt}.pt")
                torch.save(accumulated, chunk_file_path)
                print(f"[make_dataset] Saved {idx} samples → {chunk_file_path}")
                
                # Reset accumulated data and increment the counter
                accumulated = []
                cnt += 1

    def make_dataloader(
        self,
        texts: list[str],
        batch_size: int = 20,
        shuffle: bool = True,
        num_workers: int = 4,
        accelerator=None,  # ← pass your Accelerator
    ) -> DataLoader:
        # 1) tokenize each string → list[dict[str, Tensor]]
        ds = self.make_dataset(texts)

        # 2) build (or not) a DistributedSampler
        sampler = None  # No DistributedSampler since we aren't in a distributed environment
        shuffle_flag = shuffle

        # 3) same pad-to-longest collate
        def collate(batch):
            keys = batch[0].keys()
            max_len = max(len(b["input_ids"]) for b in batch)
            out = {}
            for k in keys:
                pad_val = 0 if k != "labels" else -100
                out[k] = torch.stack(
                    [F.pad(b[k], (0, max_len - len(b[k])), value=pad_val)
                     for b in batch],
                    dim=0,
                )
            return out

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle_flag,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=True,
        )

        # 4) send loader to the right device once
        if accelerator is not None:
            loader = accelerator.prepare(loader)

        return loader


    # ---------- internal -------------------------------------------
    def _encode_one(self, text: str):
        """
        Returns a dict with:
        input_ids, attention_mask, labels,
        num_embed, num_target, num_mask
        where everything up through the prefix (as decided by get_prefix_text)
        is masked (labels = -100, num_target/mask = 0).
        """
        # 1) Replace numbers and capture raw floats
        proc_text, numbers = self.replace_nums(text)

        # 2) Build the prefix string and count its tokens
        prefix_text = get_prefix_text(text)
        prefix_ids  = self.tok(prefix_text, add_special_tokens=True)["input_ids"]
        prefix_len  = len(prefix_ids)

        # 3) Tokenize the full processed text
        enc = self.tok(
            proc_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        ids  = enc.input_ids[0]          # (L,)
        attn = enc.attention_mask[0]     # (L,)
        L    = ids.size(0)

        # 4) Build shifted labels for CE
        labels = ids.clone()
        labels[:-1] = ids[1:]
        labels[-1]  = -100               # ignore last

        # 5) Prepare side‐arrays for numeric injection & regression
        num_embed  = torch.zeros(L, device=ids.device)
        num_target = torch.zeros(L, device=ids.device)
        num_mask   = torch.zeros(L, device=ids.device)

        # inject raw floats where input==<NUM>
        ptr = 0
        for i, tid in enumerate(ids):
            if tid == self.num_id:
                try:
                    num_embed[i] = float(numbers[ptr])
                    ptr += 1
                except:
                    ptr += 1
                    continue

        # mark regression where label==<NUM>
        ptr = 0
        for i, tid in enumerate(labels):
            if tid == self.num_id:
                try:
                    num_target[i] = float(numbers[ptr])
                    num_mask[i]   = 1.0
                    ptr += 1
                except:
                    ptr += 1
                    continue

        # 6) Mask out the prefix tokens from all losses
        labels[:prefix_len]     = -100
        num_target[:prefix_len] = 0.0
        num_mask[:prefix_len]   = 0.0

        return {
            "input_ids":     ids,
            "attention_mask":attn,
            "labels":        labels,
            "num_embed":     num_embed,
            "num_target":    num_target,
            "num_mask":      num_mask,
        }


    def replace_nums(self,text: str):
        """
        Replace <number> with <NUM> and return (processed_text, numbers_list).
        Non-numeric <...> remain unchanged.
        """
        numbers: List[str] = []

        def repl(m: re.Match) -> str:
            numbers.append(m.group(1))
            return "<NUM>"

        processed = self.NUM_RE.sub(repl, text)
        return processed, numbers


# Arguments for the script (you can adjust these or get them from command line arguments)
def parse_args():
    parser = argparse.ArgumentParser(description="Data preprocessing script for Qwen model")

    # Add arguments for the script
    parser.add_argument('--json_num_spec', type=str, default="data/float_encoder_columns.json", help="Path to the numeric specification JSON file")
    parser.add_argument('--num_workers_cpu', type=int, default=4, help="Number of CPU workers for DataLoader")
    parser.add_argument('--batch_size', type=int, default=2048, help="Batch size for data loading")
    parser.add_argument('--save_path', type=str, default="./data/precomputed/data_loader/processed_data.pt", help="Path to save the processed data")
    return parser.parse_args()

# Parse command-line arguments
args = parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)  # Ensure the directory exists
# Dataset to be processed
datasets = [
    "rel-amazon", "rel-avito", "rel-f1", "rel-hm", "rel-stack", "rel-trial"
]
# datasets = [
#     "rel-f1"
# ]

# List to collect all the samples
tot_samples = []

# Load the dataset
rel_ds = RelBenchRowDataset(datasets, args.json_num_spec, download=False)

# Create the DataLoader with no DistributedSampler since we are not using distributed environment
raw_loader = DataLoader(
    rel_ds,
    batch_size=8092,
    shuffle=True,  # Shuffling the data in the DataLoader
    num_workers=args.num_workers_cpu,
    collate_fn=lambda batch: batch,  # Placeholder function since no custom collate function is provided
    drop_last=False,
)

# Collect the samples into tot_samples
for i, batch_texts in tqdm(enumerate(raw_loader), total=len(raw_loader), desc="Collecting samples"):
    tot_samples.extend(batch_texts)

# Print total number of samples
print(f"Total training samples (before processing): {len(tot_samples)}")

# Initialize the tokenizer for "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained('./model/qwen_tokenizer/')  # Use the correct tokenizer for your model
# tokenizer.add_special_tokens({"additional_special_tokens": ["<NUM>"]})
# tokenizer.save_pretrained('model/qwen_tokenizer/')  # Save the updated tokenizer
processor = NumericDataProcessor(tokenizer)

# Process the collected samples into a torch Dataset
os.makedirs(args.save_path, exist_ok=True)  # Ensure the save directory exists
print(f"Processing {len(tot_samples)} samples and saving to {args.save_path}")
processed_dataset = processor.make_dataset(tot_samples,save_path=args.save_path, chunk_size=80000)
# Save the processed dataset to the specified save path


