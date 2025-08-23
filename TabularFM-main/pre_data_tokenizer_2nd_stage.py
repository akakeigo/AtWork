# pre_data_tokenizer_2nd_stage.py
# ------------------------------------------------------------
# Tokenize RelBenchTaskDataset samples (neighbors + MAIN + Q/A)
# into model-ready tensors with numeric side-channels.
#
# Outputs per-sample dict:
#   {
#     "input_ids": Tensor[L],
#     "attention_mask": Tensor[L],
#     "labels": Tensor[L],            # next-token LM; -100 masks
#     "num_embed": Tensor[L],         # floats aligned to input <NUM> tokens
#     "num_target": Tensor[L],        # floats aligned to steps predicting <NUM>
#     "num_mask": Tensor[L],          # 1 where we regress a number
#   }
#
# Design:
# - Replace every "<float>" in the text with "<NUM>", collecting the floats in order.
# - Build shifted labels; mark positions where labels == <NUM> for regression targets.
# - Mask *everything before "Answer:"* from both CE and numeric losses.
# - Save dataset in chunks; provide a DataLoader collate function (padding).
# ------------------------------------------------------------

import os
import sys
import re
import math
import glob
import json
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import argparse
from transformers import AutoTokenizer
from utils.helper_func import *
    
    
    
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except Exception as e:
    raise RuntimeError("Please `pip install transformers`") from e

    
    
OPEN_CLOSE_TAGS = [
    ("<1-hop NEIGHBOR>", "<\\1-hop NEIGHBOR>", "nei1"),
    ("<2-hop NEIGHBOR>", "<\\2-hop NEIGHBOR>", "nei2"),
    ("<MAIN>", "<\\MAIN>", "main"),
]


INF = 1e4  # large positive added as NEGATIVE after sign flip in attention

class NumericQAPreprocessor:
    r"""
    Numeric-aware tokenizer for QA-style prompts.

    Given a sample text that already contains the task-specific Q/A tail, e.g.
        "... <MAIN> ... </MAIN> Question: what is the driver's position? Answer: <-1.480576395>"
    we:
      1) Replace every numeric literal inside angle brackets with the literal token "<NUM>".
      2) Tokenize the processed text with a HuggingFace tokenizer.
      3) Build next-token labels (shifted by one; last = -100).
      4) Fill numeric side-channels:
         - num_embed[i] = float of the i-th <NUM> token in the INPUT sequence.
         - num_target[i] = float of the i-th <NUM> in the LABELS sequence where labels[i] == <NUM>.
           (i.e., step i is about to predict a <NUM> next).
         - num_mask[i] = 1.0 exactly where we wrote num_target[i].
      5) Mask everything before "Answer:" (inclusive of the "Answer:" token itself).
         That is, for tokens t with index < cutoff:
            labels[t] = -100
            num_target[t] = 0
            num_mask[t] = 0

    Notes
    -----
    • The regex only captures numerics of the form:
         <  -12,  3,  4.5,  .5,  1.,  1e-3,  -2.7E+5  >
      Non-numeric angle tags like <1-hop NEIGHBOR> are untouched.
    • Truncation:
      We let the tokenizer truncate to `max_length`. Since we only supervise
      the "Answer:" span, any truncation typically affects the context.
    """

    NUM_TOKEN = "<NUM>"
    NUM_RE = re.compile(
        r"<([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)>"
    )

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 1024):
        """
        Parameters
        ----------
        tokenizer : transformers.PreTrainedTokenizerBase
            The tokenizer to use (e.g., Qwen/Qwen2-0.5B).
            Will be augmented with "<NUM>" if missing.
        max_length : int
            Maximum sequence length for tokenization (with truncation).
        """
        self.tok = tokenizer
        self.max_len = int(max_length)

        # Ensure "<NUM>" exists
        if self.NUM_TOKEN not in self.tok.get_vocab():
            self.tok.add_special_tokens({"additional_special_tokens": [self.NUM_TOKEN]})
        # Cache the id
        self.num_id = self.tok(self.NUM_TOKEN, add_special_tokens=False)["input_ids"][0]

    # ------------------------ public API ------------------------
    def process_samples(
        self,
        samples: List[Dict[str, Any]],
        save_dir: Optional[str] = None,
        chunk_size: int = 80_000,
        verbose: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert a list of {"text": str, ...} samples into tensors.

        Parameters
        ----------
        samples : list of dict
            Each dict must contain key "text" with the full prompt including Q/A tail.
        save_dir : str or None
            If given, chunked outputs "processed_data_{k}.pt" are written here.
        chunk_size : int
            Number of samples per saved chunk.
        verbose : bool
            Log progress.

        Returns
        -------
        processed : list of dict[str, Tensor]
            The processed (possibly also saved) dataset.
        """
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        processed: List[Dict[str, torch.Tensor]] = []
        for i, ex in enumerate(tqdm(samples, disable=not verbose, desc="Tokenizing")):
            text = ex["text"]
            out = self.encode_one(text)
            processed.append(out)

            if save_dir is not None and (i + 1) % chunk_size == 0:
                part = (i + 1) // chunk_size
                path = os.path.join(save_dir, f"processed_data_{part}.pt")
                torch.save(processed[-chunk_size:], path)

        if save_dir is not None:
            # Save the full list for convenience
            torch.save(processed, os.path.join(save_dir, "processed_data_all.pt"))

        return processed


    def encode_one(self, text: str) -> Dict[str, Any]:
        """
        Encode a single QA-style sample into tensors plus segment spans.

        Returns
        -------
        {
          "input_ids":     LongTensor[L],
          "attention_mask":LongTensor[L],
          "labels":        LongTensor[L],        # next-token LM; -100 masked
          "num_embed":     FloatTensor[L],       # floats aligned to INPUT <NUM>
          "num_target":    FloatTensor[L],       # floats aligned where LABELS == <NUM>
          "num_mask":      FloatTensor[L],       # 1.0 at regression steps
          "segment_spans": List[{"kind": str, "start": int, "end": int}],
        }
        """

        # 1) Replace numeric angle-bracket literals with <NUM>, capturing values.
        proc_text, numbers = self._replace_nums(text)

        # 2) Character cutoff for masking (everything before "Answer:" is masked).
        cutoff_char = self._answer_cutoff_tokens(text)  # -1 if not found

        # 3) Tokenize the *processed* text (with <NUM>), with truncation.
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

        # 4) Next-token labels (last position ignored).
        labels = ids.clone()
        labels[:-1] = ids[1:]
        labels[-1]  = -100

        # 5) Numeric side-channels.
        num_embed  = torch.zeros(L, dtype=torch.float32)
        num_target = torch.zeros(L, dtype=torch.float32)
        num_mask   = torch.zeros(L, dtype=torch.float32)

        # 5a) num_embed: fill at INPUT positions where token == <NUM>.
        ptr_in = 0
        for i in range(L):
            if ids[i].item() == self.num_id and ptr_in < len(numbers):
                v = numbers[ptr_in]
                try:
                    num_embed[i] = float(v)
                except Exception:
                    pass
                ptr_in += 1

        # 5b) num_target/num_mask: fill at positions whose LABEL == <NUM>.
        #     (i.e., steps that are *about to* generate <NUM>)
        ptr_out = 0
        for i in range(L):
            if labels[i].item() == self.num_id and ptr_out < len(numbers):
                v = numbers[ptr_out]
                try:
                    num_target[i] = float(v)
                    num_mask[i]   = 1.0
                except Exception:
                    pass
                ptr_out += 1

        # 6) Mask everything before the answer cutoff (in *token* space).
        if cutoff_char >= 0:
            # Convert prefix (up to cutoff) to tokens AFTER numeric replacement
            # so that token indices align with `ids`.
            prefix_processed, _ = self._replace_nums(text[:cutoff_char])
            cutoff_tok = len(self.tok(prefix_processed, add_special_tokens=True)["input_ids"])

            labels[:cutoff_tok]     = -100
            num_target[:cutoff_tok] = 0.0
            num_mask[:cutoff_tok]   = 0.0

        # 7) Segment spans in token space (for block attention construction later).
        #    Spans cover the tags themselves, e.g., "<MAIN> ... <\\MAIN>".
        segment_spans = self._find_segments_token_spans(proc_text)

        return {
            "input_ids":      ids,           # LongTensor[L]
            "attention_mask": attn,          # LongTensor[L]
            "labels":         labels,        # LongTensor[L]
            "num_embed":      num_embed,     # FloatTensor[L]
            "num_target":     num_target,    # FloatTensor[L]
            "num_mask":       num_mask,      # FloatTensor[L]
            "segment_spans":  segment_spans, # list of dicts
        }


    # ------------------------ utilities ------------------------
    def _replace_nums(self, text: str) -> Tuple[str, List[str]]:
        """
        Replace <number> → <NUM>, collecting numeric strings in order.
        Non-numeric angle-bracket tags remain unchanged.
        """
        nums: List[str] = []

        def repl(m: re.Match) -> str:
            nums.append(m.group(1))
            return self.NUM_TOKEN

        processed = self.NUM_RE.sub(repl, text)
        return processed, nums

    def _answer_cutoff_tokens(self, full_text: str) -> int:
        """
        Return the character index where 'Answer:' begins; -1 if not present.
        """
        idx = full_text.find("Answer:")
        return idx if idx >= 0 else -1


    def _tok_count(self, s: str) -> int:
        return len(self.tok(s, add_special_tokens=True)["input_ids"])

    def _find_segments_token_spans(self, processed_text: str) -> list[dict]:
        """
        Returns a list of {'kind': str, 'start': int, 'end': int} in token indices
        [start, end), including the opening/closing tags inside the span.
        """
        spans = []
        for open_tag, close_tag, kind in OPEN_CLOSE_TAGS:
            start = 0
            while True:
                i = processed_text.find(open_tag, start)
                if i < 0:
                    break
                j = processed_text.find(close_tag, i + len(open_tag))
                if j < 0:
                    break
                j_end = j + len(close_tag)

                # token indices in processed space
                tok_start = self._tok_count(processed_text[:i])
                tok_end   = self._tok_count(processed_text[:j_end])
                spans.append({"kind": kind, "start": tok_start, "end": tok_end})

                start = j_end  # continue search after this block
        # sort by start
        spans.sort(key=lambda d: d["start"])
        return spans


# ------------------------ dataloader utils ------------------------
def qa_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Pad a batch of variable-length examples to the max length in the batch.

    Keys padded: input_ids (0), attention_mask (0), labels (-100),
                 num_embed (0.0), num_target (0.0), num_mask (0.0)
    """
    keys = batch[0].keys()
    max_len = max(v["input_ids"].size(0) for v in batch)
    out: Dict[str, torch.Tensor] = {}

    for k in keys:
        pad_val = -100 if k == "labels" else 0
        dtype   = batch[0][k].dtype
        padded  = []
        for ex in batch:
            t = ex[k]
            if t.size(0) < max_len:
                pad = torch.full((max_len - t.size(0),), pad_val, dtype=dtype)
                t   = torch.cat([t, pad], dim=0)
            padded.append(t)
        out[k] = torch.stack(padded, dim=0)
    return out


def _build_block_causal_mask(L: int,
                             spans: list[dict],
                             valid_len: int) -> torch.Tensor:
    """
    Returns (L, L) float mask: 0.0 allowed, -INF blocked.
    'valid_len' is the true length (before padding).
    """
    allow = torch.zeros(L, L, dtype=torch.bool)

    # map each token to a segment id
    seg_id = torch.full((L,), fill_value=-1, dtype=torch.long)
    # assign IDs: 0..K-1 to neighbors, MAIN gets id = 10_000 (special)
    cur = 0
    for sp in spans:
        a, b = sp["start"], sp["end"]
        a = max(0, min(a, L)); b = max(0, min(b, L))
        if a >= b:
            continue
        if sp["kind"] == "main":
            seg_id[a:b] = 10_000
        else:
            seg_id[a:b] = cur
            cur += 1

    # default: self-only for tokens not in any span
    allow |= torch.eye(L, dtype=torch.bool)

    # MAIN: full attention (will be intersected with causal & padding)
    main_mask = (seg_id == 10_000)
    if main_mask.any():
        allow[main_mask, :] = True

    # each neighbor block: block-diagonal attention
    for k in range(cur):
        block = (seg_id == k)
        if block.any():
            idx = block.nonzero(as_tuple=True)[0]
            allow[idx.unsqueeze(1), idx.unsqueeze(0)] = True

    # causality (strictly lower-tri incl diag)
    tril = torch.tril(torch.ones(L, L, dtype=torch.bool))
    allow &= tril

    # cut keys beyond valid_len (padding)
    if valid_len < L:
        key_valid = torch.zeros(L, dtype=torch.bool)
        key_valid[:valid_len] = True
        allow &= key_valid.view(1, L).expand(L, L)

    # convert to additive bias
    attn_bias = torch.zeros(L, L, dtype=torch.float32)
    attn_bias[~allow] = -INF
    return attn_bias



def qa_collate_fn_with_block_mask(batch):
    """
    Pads tensors and builds per-sample (1,L,L) attention bias.
    """
    keys = [k for k in batch[0].keys() if k != "segment_spans"]
    Lmax = max(ex["input_ids"].size(0) for ex in batch)

    out = {}
    for k in keys:
        pad_val = -100 if k == "labels" else 0
        dtype   = batch[0][k].dtype
        padded  = []
        for ex in batch:
            t = ex[k]
            if t.size(0) < Lmax:
                pad = torch.full((Lmax - t.size(0),), pad_val, dtype=dtype)
                t   = torch.cat([t, pad], dim=0)
            padded.append(t)
        out[k] = torch.stack(padded, dim=0)

    # build attn_bias
    biases = []
    for ex in batch:
        L = ex["input_ids"].size(0)
        spans = ex["segment_spans"]
        bias = _build_block_causal_mask(Lmax, spans, valid_len=L)
        biases.append(bias.unsqueeze(0))   # (1,Lmax,Lmax)
    out["attn_bias"] = torch.stack(biases, dim=0)  # (B,1,Lmax,Lmax)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JSONL text_input to token_input for one dataset/task across specified splits."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Directory of a single dataset/task, i.e., .../text_input/<dataset>/<task>",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/sunmingyang/xiaomin/lxr/TabularFM/data/precomputed_2nd_stage/token_input/",
        help="Root dir to write processed_data_*.pt under <dataset>/<task>/<split>/",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="/sunmingyang/xiaomin/lxr/TabularFM/model/qwen_tokenizer/",
        help="HF tokenizer path/name (will be augmented with <NUM> if missing).",
    )
    parser.add_argument("--max_length", type=int, default=1596, help="Default max tokens (train uses 1596; non-train uses 5092).")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=40_000,
        help="How many samples per processed_data_*.pt chunk.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process (directories must be named exactly like these).",
    )
    # --filter is no longer needed; kept only for backward compatibility
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Deprecated: ignored in dataset/task mode.",
    )
    args = parser.parse_args()

    # 1) Resolve dataset/task identity from the input_root path
    ds_task_root = os.path.abspath(args.input_root.rstrip(os.sep))
    comps = ds_task_root.split(os.sep)
    dataset = comps[-2] if len(comps) >= 2 else "unknown_dataset"
    task    = comps[-1] if len(comps) >= 1 else "unknown_task"

    # 2) Validate splits exist under this dataset/task root
    candidate_splits = [s for s in args.splits if os.path.isdir(os.path.join(ds_task_root, s))]
    if not candidate_splits:
        cprint(f"No split directories {args.splits} were found under {ds_task_root}.")
        sys.exit(0)

    # 3) Load tokenizer once
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # 4) Process each split directory directly (no global os.walk)
    os.makedirs(args.output_root, exist_ok=True)
    total_dirs = 0

    for split in candidate_splits:
        dirpath = os.path.join(ds_task_root, split)

        # Per-split token budget
        split_max_len = 1596 if split == "train" else 5092
        pre = NumericQAPreprocessor(tokenizer=tok, max_length=split_max_len)
        cprint(f"\n[{dataset}/{task}/{split}] max_length={split_max_len}")

        # Collect all jsonl files under this split
        jsonl_files = sorted(glob.glob(os.path.join(dirpath, "*.jsonl")))
        if not jsonl_files:
            cprint(f"[skip] No JSONL files in {dirpath}")
            continue

        out_dir = os.path.join(args.output_root, dataset, task, split)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # os.makedirs(out_dir, exist_ok=True)
        cprint(f"[{dataset}/{task}/{split}] found {len(jsonl_files)} file(s) → {out_dir}")

        chunk_idx = 1
        saved = 0
        skipped = 0
        buf: list[dict] = []
        cprint("all json files:")
        for j in jsonl_files:
            cprint(j)

        for file_path in tqdm(jsonl_files, desc="loading jsonl files",
                              unit="file", position=0, leave=True, total=len(jsonl_files)):
            fname = os.path.basename(file_path)
            tqdm.write(f"  - reading {fname}")

            # 1st pass: count lines to enable a proper %/ETA
            with open(file_path, "r", encoding="utf-8") as f:
                n_lines = sum(1 for _ in f)

            # 2nd pass: actual processing with a % bar
            with open(file_path, "r", encoding="utf-8") as f, \
                 tqdm(total=n_lines, desc=f"    encoding ({fname})",
                      unit="line", position=1, leave=False, dynamic_ncols=True) as pbar:
                for line in f:
                        pbar.update(1)
                    # try:
                        obj = json.loads(line)
                        text = obj.get("text", None)
                        if not text:
                            skipped += 1
                            continue

                        ex = pre.encode_one(text)  # dict of tensors + segment_spans
                        buf.append(ex)

                        if len(buf) >= args.chunk_size:
                            out_path = os.path.join(out_dir, f"processed_data_{chunk_idx}.pt")
                            torch.save(buf, out_path)
                            saved += len(buf)
                            buf.clear()
                            chunk_idx += 1

                    # except Exception:
                        skipped += 1
                        # continue

        # Final flush after all files
        if buf:
            out_path = os.path.join(out_dir, f"processed_data_{chunk_idx}.pt")
            torch.save(buf, out_path)
            saved += len(buf)
            buf.clear()

        cprint(f"  → saved {saved} samples in {chunk_idx} chunk(s); skipped {skipped} bad line(s).")
        total_dirs += 1

    cprint(f"\nDone. Processed {total_dirs} split directories into {args.output_root}.")
