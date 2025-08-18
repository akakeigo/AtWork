"""
numeric_qwen.py
────────────────
Fine‑tune a Qwen model so that the special token <NUM> is predicted
with (1) cross‑entropy in the token stream  and  (2) an MSE regression
head that outputs the actual float value.

Requires:
  pip install torch datasets transformers==4.41 openai tqdm
"""
import os
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["TRANSFORMERS_NO_META_DEVICE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import random
import re
import math
from typing import List, Dict, Any, Optional
from termcolor import colored
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import  DistributedSampler, ConcatDataset
from accelerate import Accelerator,DistributedDataParallelKwargs
import argparse
from transformers import (
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
    AutoConfig
)

from torch.distributed import ReduceOp
import torch.distributed as dist

from tqdm.auto import tqdm
from termcolor import colored
from floatEncoder import SignedFloatAutoencoder
from PretrainingDataset import RelBenchRowDataset
import wandb
import warnings
import glob
import os
warnings.filterwarnings("ignore")

# helper

def load_distributed_data(file_dir: str, rank: int, world_size: int):
    """
    Load data files distributed across ranks.
    Each rank loads a subset of the total files.
    """
    
    # Scan directory for processed_data_*.pt files
    pattern = os.path.join(file_dir, "processed_data_*.pt")
    all_files = glob.glob(pattern)
    
    # Sort files by the numeric part (1, 2, 3, ...)
    def extract_number(filename):
        base = os.path.basename(filename)
        # Extract number from processed_data_XXX.pt
        num_str = base.replace("processed_data_", "").replace(".pt", "")
        return int(num_str)
    
    all_files.sort(key=extract_number)
    
    if rank == 0:
        print(f"Found {len(all_files)} data files in {file_dir}")
    
    # Distribute files across ranks
    files_per_rank = len(all_files) // world_size
    remainder = len(all_files) % world_size
    
    # Calculate start and end indices for this rank
    if rank < remainder:
        # First 'remainder' ranks get one extra file
        start_idx = rank * (files_per_rank + 1)
        end_idx = start_idx + files_per_rank + 1
    else:
        # Remaining ranks get the base number of files
        start_idx = rank * files_per_rank + remainder
        end_idx = start_idx + files_per_rank
    
    # Get files for this rank
    rank_files = all_files[start_idx:end_idx]
    # rank_files = rank_files[:2] #! for test
    if rank == 0:
        print(f"[Rank {rank}] Loading files {start_idx} to {end_idx-1}: {[os.path.basename(f) for f in rank_files]}")
    
    # Load tensors from assigned files
    all_tensors = []
    cnt = 0
    for file_path in tqdm(rank_files,desc=f"Loading files"):
        if os.path.exists(file_path):
            loaded_data = torch.load(file_path, map_location="cpu")
            
            if isinstance(loaded_data, list):
                all_tensors.extend(loaded_data)
            else:
                print(f"[Rank {rank}] Warning: Data in {file_path} is not a list. Skipping.")
        else:
            print(f"[Rank {rank}] File not found: {file_path}")
        # cnt += 1
        # if cnt>5: break
    
    print(f"[Rank {rank}] Loaded {len(all_tensors)} samples total")
    return all_tensors
    

def compute_per_sample_loss(
    logits: torch.FloatTensor,   # (B, L, V)
    labels: torch.LongTensor,    # (B, L), with ignore_index = -100
    preds: torch.Tensor,         # (B, L), numeric head outputs
    num_target: torch.Tensor,    # (B, L) raw float targets
    num_mask: torch.Tensor,      # (B, L), 1 where we regress
    ce_weight: float = 1.0,
    mse_weight: float = 1.0,
):
    B, L, V = logits.size()
    logit_ce = logits.transpose(1, 2)
    # --- 1) Cross‐Entropy per token & per sample ---
    ce_tok = F.cross_entropy(
    logit_ce,
    labels,
    reduction="none",
    ignore_index=-100
)             # back to (B,L)                                  # (B, L)

    # count valid CE tokens per sample
    valid_ce = (labels != -100).sum(dim=1).clamp(min=1)  # (B,)
    ce_per_sample = ce_tok.sum(dim=1) / valid_ce        # (B,)

    # --- 2) Numeric MSE per sample ---
    # squared error, zeroed where mask=0
    se = num_mask * (preds - num_target) ** 2           # (B, L)
    valid_num = num_mask.sum(dim=1).clamp(min=1)        # (B,)
    mse_per_sample = se.sum(dim=1) / valid_num          # (B,)

    # --- 3) Total per sample ---
    total_per_sample = ce_weight * ce_per_sample + mse_weight * mse_per_sample

    return total_per_sample, ce_per_sample, mse_per_sample


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

# ---------------------------------------------------------------------
# 1)  Data processing
# ---------------------------------------------------------------------
class NumericDataset(Dataset):
    """Holds already‑tokenised samples."""

    def __init__(self, samples: List[Dict[str, torch.Tensor]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class NumericDataProcessor:
    r"""
    Converts raw strings with angle‑bracket numbers into the tensors
    needed by the Numeric LLaMA model.

    • Any numeric literal wrapped in “< … >” is replaced by literal token
      “<NUM>”  (tokenised as a single ID).
    • Side‑arrays keep the float values and masks.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 256,
    ):
        self.tok = tokenizer
        self.max_len = max_length
        # convenience: fetch <NUM> id once
        self.num_id = self.tok("<NUM>")["input_ids"][0]
        self.NUM_RE = re.compile(
        r"<([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)>"
    )

    def make_dataset(self, texts: List[str], accelerator) -> NumericDataset:
        """Return torch Dataset of tokenised training examples."""
        if accelerator.is_main_process:  # Only show the progress bar for rank 0
            samples = [self._encode_one(t) for t in tqdm(texts, desc="Tokenizing texts")]
        else:
            samples = [self._encode_one(t) for t in texts]  # No progress bar for other ranks
        return NumericDataset(samples)
    
    
    
    def make_dataloader(
        self,
        texts: list[str],
        batch_size: int = 20,
        shuffle: bool = True,
        num_workers: int = 4,
        accelerator=None,
        precomputed_tensors: Optional[List[torch.Tensor]] = None,
        use_distributed_sampler: bool = True,  # New parameter
    ) -> DataLoader:
        # 1) Create dataset
        if precomputed_tensors is not None:
            ds = NumericDataset(precomputed_tensors)
        else:
            ds = self.make_dataset(texts, accelerator)

        # 2) Determine if we need DistributedSampler
        # When data is pre-sharded across ranks, we typically don't use DistributedSampler
        # Only use it when all ranks have the same full dataset
        sampler = None
        if accelerator.num_processes > 1 and use_distributed_sampler and not precomputed_tensors:
            sampler = DistributedSampler(ds, shuffle=shuffle)
            shuffle_flag = False
        else:
            shuffle_flag = shuffle

        # 3) Collate function for padding
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

        # 4) Create DataLoader
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle_flag,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=True,  # Important for distributed training
        )
        # 5) Prepare with accelerator
        # if accelerator is not None:
        #     loader = accelerator.prepare(loader)
        return loader

    # ---------- internal -------------------------------------------
    def _encode_one(self, text: str) -> Dict[str, torch.Tensor]:
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
        # Subtract 1 if your tokenizer adds a BOS token you don't want to count.
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


# ---------------------------------------------------------------------
# 2)  Model wrapper
# ---------------------------------------------------------------------
# 3) In the numeric‐wrapper, replace llama‐specific calls:

class QwenNumericModel(nn.Module):
    def __init__(self, base_name="Qwen/Qwen2-0.5B", float_encoder=None):
        super().__init__()
        # Download from HF Hub (no local path needed)
        config = AutoConfig.from_pretrained(
            base_name,
            trust_remote_code=True,
            local_files_only=True
        )
        config.use_flash_attention_2 = True
        config.output_hidden_states = True

        self.qwen = AutoModelForCausalLM.from_pretrained(
            base_name,
            trust_remote_code=True,
            config=config,
            local_files_only=True
        )

        self.float_encoder = float_encoder
        d_model = self.qwen.config.hidden_size
        self.num_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        # plug‑in float encoder
        self.float_encoder = float_encoder

    # ------------------------------------------------------------
    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        num_embed,      # raw floats  (B,L)
        num_target,
        num_mask,
        ce_weight=1.0,
        mse_weight=1.0,
    ):
        # ------ (1) base token embeddings -----------------------
        B, L = num_embed.shape
        d_model = self.qwen.config.hidden_size
        tok_emb = self.qwen.get_input_embeddings()(input_ids)         # (B,L,H)

        # ------ (2) numeric embedding injection ----------------
        mask_inp = (num_embed != 0)                                   # (B,L) bool
        if mask_inp.any():
            # flatten indices of non-zero numbers
            nz_idx = mask_inp.nonzero(as_tuple=False)                 # (N, 2) -> [ [b,l], ... ]
            vals   = num_embed[mask_inp].unsqueeze(-1)                # (N,1)

            # encode only those values
            enc_vec = self.float_encoder.encode(vals)                        # (N,H)
            enc_vec = enc_vec.squeeze(1)
            enc_vec = enc_vec.to(tok_emb.dtype)

            # create a zeros tensor and scatter
            num_vec = torch.zeros(B, L, d_model, device=tok_emb.device, dtype=tok_emb.dtype)
            num_vec[nz_idx[:,0], nz_idx[:,1]] = enc_vec               # put back

            tok_emb = tok_emb + num_vec

        # ------ (3) run transformer & CE loss ------------------
        out = self.qwen(
            inputs_embeds=tok_emb,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = out.logits              # (B,L,V)
        hidden = out.hidden_states[-1]   # (B,L,H)
        preds = self.num_head(hidden).squeeze(-1)   # (B,L)

        total, ce_ps, mse_ps = compute_per_sample_loss(
            logits, labels, preds, num_target, num_mask, ce_weight, mse_weight
        )
        # print("forward stats:", ce_ps, mse_ps)

        return total, ce_ps, mse_ps

        # # ------ (4) numeric MSE loss ---------------------------
        # preds     = self.num_head(hidden).squeeze(-1)          # (B,L)
        # mse       = (num_mask * (preds - num_target) ** 2).sum()
        # mse_loss  = mse / num_mask.sum().clamp(min=1.0)

        # total = ce_weight * ce_loss + mse_weight * mse_loss
        # return total, ce_loss.detach(), mse_loss.detach()

  

    @torch.no_grad()
    def generate_with_numbers(self, prompt: str, tokenizer, max_new_tokens=30, greedy=True, format_float=lambda x: f"{x:.2f}"):
        self.qwen.eval()
        device = next(self.parameters()).device

        num_id = tokenizer("<NUM>")["input_ids"][0]
        eos_id = tokenizer.eos_token_id

        # encode prompt
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device).squeeze(0).tolist()
        num_embed = [0.0] * len(ids)

        generated, numeric_values = tokenizer.decode(ids, skip_special_tokens=True), []
        for _ in range(max_new_tokens):
            ids_t = torch.tensor([ids], device=device)
            num_e = torch.tensor([num_embed], device=device)
            attn = torch.ones_like(ids_t)

            # inject numeric embedding
            tok_emb = self.qwen.get_input_embeddings()(ids_t)
            mask_in = (num_e != 0).unsqueeze(-1)
            tok_emb = tok_emb + mask_in * num_e.unsqueeze(-1)

            out = self.qwen(inputs_embeds=tok_emb, attention_mask=attn, output_hidden_states=True)
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1).item() if greedy else torch.multinomial(logits.softmax(-1), 1).item()

            ids.append(next_id); num_embed.append(0.0)
            if next_id == eos_id:
                break

            if next_id == num_id:
                h_last = out.hidden_states[-1][:, -1]           # (1,H)
                v = self.num_head(h_last).item()
                num_embed[-1] = v
                numeric_values.append(v)
                generated += f"<{format_float(v)}>"
            else:
                generated += tokenizer.decode([next_id], skip_special_tokens=True)

        return generated, numeric_values


# ---------------------------------------------------------------------
# 3)  Trainer
# ---------------------------------------------------------------------
class QwenNumericTrainer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        optimizer: torch.optim.Optimizer,
        accelerator: Accelerator,
        scheduler,                   # <-- new
        grad_accum_steps: int = 1,   # <-- new
    ):
        self.model       = model
        self.tok         = tokenizer
        self.optimizer   = optimizer
        self.accelerator = accelerator
        self.scheduler   = scheduler
        self.grad_accum  = grad_accum_steps


    def fit_epoch(self, dataloader: DataLoader, updates_per_epoch: int, epoch: int):
        """
        Runs one epoch with DDP-safe control flow and returns averaged losses.

        Inputs
        -------
        dataloader : torch.utils.data.DataLoader
            The per-rank DataLoader (already sharded or not). Must be finite.
        updates_per_epoch : int
            Number of optimizer steps per epoch (for logging cadence). Typically
            ceil(len(dataloader) / grad_accum_steps) computed on the smallest rank.
        epoch : int
            1-based epoch index used for checkpoint naming.

        Outputs
        -------
        avg_total : float
            Average total loss across optimizer steps in this epoch.
        avg_ce    : float
            Average cross-entropy term across optimizer steps.
        avg_mse   : float
            Average numeric MSE term across optimizer steps.

        Notes
        -----
        1) All ranks iterate exactly the same number of batches (min across ranks).
        2) A synchronized pre-check guarantees that either all ranks enter forward
        or all ranks skip the batch, preventing stragglers inside DDP collectives.
        3) If any rank OOMs during forward, all ranks synchronously skip that batch.
        """

        def _allreduce_min_int(x: int) -> int:
            t = torch.tensor([int(x)], device=self.accelerator.device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(t, op=ReduceOp.MIN)
            return int(t.item())

        def _allreduce_max_int(x: int) -> int:
            t = torch.tensor([int(x)], device=self.accelerator.device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(t, op=ReduceOp.MAX)
            return int(t.item())

        self.model.train()
        is_main = self.accelerator.is_main_process

        tot_loss = 0.0
        tot_ce   = 0.0
        tot_mse  = 0.0
        steps    = 0  # optimizer steps counted (after accumulation)


        # Four evenly spaced optimizer-step milestones within this epoch
        save_points = sorted(set(
            max(1, int(round(i * updates_per_epoch / 4.0)))
            for i in (1, 2, 3, 4)
        ))
        saved_points = set()

        # Ensure identical number of batches per rank
        local_steps = len(dataloader)
        max_common_steps = _allreduce_min_int(local_steps)
        it = iter(dataloader)

        log_every = max(1, updates_per_epoch // 6)

        pbar = tqdm(range(max_common_steps), desc="Training", disable=not is_main)
        for _ in pbar:
            # 0) Fetch batch on every rank
            batch = next(it)
            # move to device non_blocking (DataLoader uses pin_memory=True)
            batch = {k: v.to(self.accelerator.device, non_blocking=True) for k, v in batch.items()}

            # 1) Synchronized PRE-CHECK before touching the model
            #    Proceed only if every rank has some token to train CE or some numeric to regress.
            has_valid_local = int(((batch["labels"] != -100).any() or (batch["num_mask"] != 0).any()).item())
            proceed = _allreduce_min_int(has_valid_local)  # 1 iff all ranks have valid data
            if proceed == 0:
                # Everyone skips this batch without entering forward/backward
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # 2) Forward; if any rank OOMs, all ranks skip together
            saw_oom_local = 0
            try:
                total_ps, ce_ps, mse_ps = self.model(**batch)  # per-sample losses (B,)
            except torch.cuda.OutOfMemoryError:
                saw_oom_local = 1

            if _allreduce_max_int(saw_oom_local) > 0:
                # Synchronized skip on OOM
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

            # 3) Compute scalar losses on valid samples (avoid NaNs/Infs)
            valid = torch.isfinite(total_ps)
            if int(valid.sum().item()) == 0:
                # Extremely rare if pre-check passed; still sync-skip to be safe
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss    = total_ps[valid].mean()
            ce_loss = ce_ps[valid].mean()
            ms_loss = mse_ps[valid].mean()

            # 4) Backward + step under Accelerate accumulation gate
            with self.accelerator.accumulate(self.model):
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    # Only on the micro-step that triggers an optimizer step
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            # 5) Logging and running means only when an optimizer step actually occurred
            if self.accelerator.sync_gradients:
                steps += 1
                tot_loss += float(loss.detach().item())
                tot_ce   += float(ce_loss.detach().item())
                tot_mse  += float(ms_loss.detach().item())

                if is_main and steps % log_every == 0:
                    wandb.log({
                        "steps": steps,
                        "loss/total": tot_loss / steps,
                        "loss/ce":    tot_ce   / steps,
                        "loss/mse":   tot_mse  / steps,
                        "lr": self.scheduler.get_last_lr()[0],
                    })

                if is_main and steps > 0:
                    pbar.set_postfix(
                        loss=tot_loss / steps,
                        ce=tot_ce / steps,
                        mse=tot_mse / steps,
                        lr=self.scheduler.get_last_lr()[0],
                    )

                # 5b) Save 4 evenly spaced checkpoints per epoch (DDP-safe)
                if (steps in save_points) and (steps not in saved_points):
                    save_dir = os.path.join("firstStagePretraining", f"epoch_{epoch}_steps_{steps:06d}")
                    if is_main:
                        os.makedirs(save_dir, exist_ok=True)
                    # ensure all ranks participate in the same save
                    self.accelerator.wait_for_everyone()
                    self.accelerator.save_state(save_dir)
                    self.accelerator.wait_for_everyone()
                    if is_main:
                        print(f"[epoch {epoch}] saved checkpoint at optimizer step {steps} → {save_dir}")
                    saved_points.add(steps)

        if steps == 0:
            return float("nan"), float("nan"), float("nan")
        return tot_loss / steps, tot_ce / steps, tot_mse / steps


# -------------- main --------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--float_encoder_ckpt", default="model/float_encoder.pt")
    parser.add_argument("--json_num_spec", default="data/float_encoder_columns.json")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--scheduler_type", default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model_path", default="checkpoints/qwen_numeric")
    parser.add_argument("--wandb_project", default="rel-LLM-1st-Stage")
    parser.add_argument("--wandb_run", default="rel-LLM-run")
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--gather_bs", type=int, default=1024)
    parser.add_argument("--num_workers_cpu", type=int, default=2)
    parser.add_argument("--download_relbench", action="store_true")
    # parser.add_argument("--datasets", nargs="+", default=[
    #     "rel-amazon", "rel-avito", "rel-f1", "rel-hm", "rel-stack", "rel-trial"
    # ])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ddp_kwargs = DistributedDataParallelKwargs(
        broadcast_buffers=False,
        find_unused_parameters=False,
        static_graph=True,   
    )
    accelerator = Accelerator(
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=args.grad_accum_steps,
    )
    print(f"[rank {accelerator.process_index}] assigned device: {accelerator.device}", flush=True)
    is_main = accelerator.is_main_process
    device = accelerator.device

    # ----- logging / dirs -----
    if is_main:
        os.makedirs(args.model_path, exist_ok=True)
        wandb.init(project=args.wandb_project, name=args.wandb_run,
                   config={"stage": "pretraining_tables"})
    
    # ----- tokenizer (identical on all ranks) -----
    # tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    # tok.add_special_tokens({"additional_special_tokens": ["<NUM>"]})
    print ("loading tok")
    tok = AutoTokenizer.from_pretrained('./model/qwen_tokenizer/') #! load from precomputed tokenizer
    vocab_size = len(tok)
    # accelerator.wait_for_everyone()

    # ----- float encoder -----
    float_encoder = SignedFloatAutoencoder(
        d_model=896, d_mant=896 // 2, emin=-20, emax=20,
        predict_log=True, lambda_sign=1.0, base=2
    )
    float_encoder.load_state_dict(torch.load(args.float_encoder_ckpt, map_location="cpu"))
    print ("loading float enc")
    for p in float_encoder.parameters():
        p.requires_grad = False
    float_encoder.eval()
    float_encoder.to(accelerator.device)
    processor = NumericDataProcessor(tok, max_length=args.max_length)
    # ----- model -----
    print ("loading qwen")
    model_name = "Qwen/Qwen2-0.5B"
    if accelerator.is_main_process:
        # trigger a single download to the shared cache
        _ = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        _ = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, low_cpu_mem_usage=True
        )
    accelerator.wait_for_everyone()
    
    model = QwenNumericModel(base_name=args.model_name, float_encoder=float_encoder)
    model.qwen.resize_token_embeddings(vocab_size)
    model.to(accelerator.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # prepare model/optimizer now (scheduler later)

    for i, p in enumerate(model.parameters()):
        if not p.device.type.startswith("cuda"):
            print(f"[rank {accelerator.process_index}] param {i} is on {p.device}", flush=True)
    try:
        model = accelerator.prepare(model)
        optimizer = accelerator.prepare(optimizer)
    except Exception as e:
        import traceback
        accelerator.print(f"[rank {accelerator.process_index}] prepare() failed: {e}")
        traceback.print_exc()
        raise

    scheduler = None  # will be created after first epoch's loader size

    trainer = None  # will be built after scheduler

    # Make sure it is deterministic across ranks → use the same seed per epoch.

    # datasets = [
    #     "rel-f1",
    # ]
    
    
    # Get rank and world size from accelerator
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    
    
    # ------------- training loop -------------
    for ep in range(args.epochs):
        print(f"\n===== Epoch {ep+1}/{args.epochs} =====")

        #! load data loader every epoch
        # Define the data directory
        data_dir = f"./data/precomputed/epoch{int(ep+1)}/"  # Adjust this path as needed

        # Load distributed data for this rank
        data_tensors = load_distributed_data(data_dir, rank, world_size)
        dloader = processor.make_dataloader(
            [],  # Empty list since we're using precomputed_tensors
            batch_size=args.batch_size,
            shuffle=True,  # Shuffle within each rank's subset
            num_workers=args.num_workers_cpu,
            accelerator=accelerator,
            precomputed_tensors=data_tensors
        )
        
        if scheduler is None:
            steps_per_epoch = len(dloader)
            print (f"[rank {accelerator.process_index}] steps_per_epoch: {steps_per_epoch}")
            updates_per_epoch = math.ceil(steps_per_epoch / args.grad_accum_steps)
            print(f"[rank {accelerator.process_index}] updates_per_epoch: {updates_per_epoch}")
            total_updates = updates_per_epoch * args.epochs
            warmup_updates = int(args.warmup_ratio * total_updates)
            scheduler = get_scheduler(
                name=args.scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=warmup_updates,
                num_training_steps=total_updates,
            )
            scheduler = accelerator.prepare(scheduler)
        
        accelerator.wait_for_everyone()

        trainer = QwenNumericTrainer(
            model=model,
            tokenizer=tok,
            optimizer=optimizer,
            accelerator=accelerator,
            scheduler=scheduler,
            grad_accum_steps=args.grad_accum_steps,
        )

        # === Train one epoch ===
        avg_loss, avg_ce, avg_mse = trainer.fit_epoch(dloader,updates_per_epoch= updates_per_epoch,epoch=ep+1)
        if is_main:
            print(f"Epoch {ep+1} completed - Loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, MSE: {avg_mse:.4f}")
    if is_main:
        wandb.finish()

if __name__ == "__main__":
    main()
    
    
    
    
