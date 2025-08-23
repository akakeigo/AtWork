import os
import sys
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
from torch.utils.data import Dataset
from relbench.datasets import get_dataset
from torch.utils.data import DataLoader
from relbench.tasks import get_task
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.testing.text_embedder import HashTextEmbedder
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import argparse
# from pre_data_tokenizer_2nd_stage import NumericQAPreprocessor
SEP = " [SEP] "
MAIN_OPEN, MAIN_CLOSE = "<MAIN>", "<\\MAIN>"
NEI1_OPEN, NEI1_CLOSE = "<1-hop NEIGHBOR>", "<\\1-hop NEIGHBOR>"
NEI2_OPEN, NEI2_CLOSE = "<2-hop NEIGHBOR>", "<\\2-hop NEIGHBOR>"


class RelBenchTaskDataset(Dataset):
    """
    Build task-specific textual context from RelBench with token-budgeted neighbors.

    Behavior:
      1) Collect up to `per_table_k` 1-hop neighbors per table.
      2) If header + 1-hop + MAIN exceeds `max_length` tokens, trim 1-hop neighbors and STOP.
      3) Else (fits), optionally add 2-hop parents of the chosen child rows until budget is reached.
    """

    def __init__(
        self,
        dataset_name: str,
        task_name: str,
        split: str = "train",
        numeric_columns_spec: Dict = None,
        download: bool = True,
        device: str = "cpu",
        per_table_k: int = 3,                 # neighbors per child table
        include_temporal_context: bool = True,
        include_two_hop_parents: bool = True, # add 2-hop parents if budget permits
        tokenizer=None,                       # pass a HuggingFace tokenizer (e.g., Qwen/Qwen2-0.5B)
        max_length: int = 1024,               # token budget
        tokenizer_name_if_none: str = "Qwen/Qwen2-0.5B",
    ):
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.split = split
        self.per_table_k = per_table_k
        self.include_temporal_context = include_temporal_context
        self.include_two_hop_parents = include_two_hop_parents
        self.max_length = int(max_length)

        # Tokenizer (used only for budgeting at dataset time)
        if tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_if_none)
        else:
            self.tokenizer = tokenizer

        # Load dataset and task
        self.dataset = get_dataset(name=dataset_name, download=download)
        self.task = get_task(dataset_name, task_name, download=download)
        self.task_table = self.task.get_table(split) if split != "test" \
            else self.task.get_table(split, mask_input_cols=False)

        # DB and hetero graph
        self.db = self.dataset.get_db()
        col_to_stype_dict = get_stype_proposal(self.db)
        text_embedder_cfg = TextEmbedderConfig(text_embedder=HashTextEmbedder(16), batch_size=None)
        self.data, self.col_stats_dict = make_pkey_fkey_graph(
            self.db, col_to_stype_dict=col_to_stype_dict, text_embedder_cfg=text_embedder_cfg
        )

        # Numeric columns spec
        self.numeric_columns_spec = numeric_columns_spec or {}

        # Task meta
        self.task_type = self.task.task_type
        self.entity_table = self.task.entity_table
        self.entity_col = self.task.entity_col
        self.target_col = self.task.target_col
        self.time_col = self.task.time_col

        # Task rows
        self.entity_ids = self.task_table.df[self.entity_col].values
        self.timestamps = self.task_table.df[self.time_col].values if self.time_col else None
        self.labels = self.task_table.df[self.target_col].values

        # PK
        self.entity_pkey = self.db.table_dict[self.entity_table].pkey_col

        # Regression stats
        if self.task_type.name == "REGRESSION":
            valid = self.labels[~np.isnan(self.labels)]
            self.label_mean = float(valid.mean()) if len(valid) else 0.0
            self.label_std  = float(valid.std())  if len(valid) else 1.0
            if not np.isfinite(self.label_std) or self.label_std == 0:
                self.label_std = 1.0
        else:
            self.label_mean = 0.0
            self.label_std  = 1.0

        # Build edge maps for 1-hop and 2-hop traversal
        self._build_neighbor_maps()

    # ---------- graph maps ----------
    def _build_neighbor_maps(self):
        # child (src) -> list of (fkey_col_in_src, parent_table)
        self.forward_edges_by_src = defaultdict(list)
        # parent (src in rev-edge) -> list of (fkey_col_in_child, child_table)
        self.reverse_edges_by_parent = defaultdict(list)

        for (src, edge_name, dst) in self.data.edge_types:
            if edge_name.startswith("f2p_"):
                fkey = edge_name[len("f2p_"):]
                self.forward_edges_by_src[src].append((fkey, dst))
            elif edge_name.startswith("rev_f2p_"):
                fkey = edge_name[len("rev_f2p_"):]
                # here 'src' is the parent table
                self.reverse_edges_by_parent[src].append((fkey, dst))

    # ---------- dataset protocol ----------
    def __len__(self) -> int:
        return len(self.entity_ids)


    # --- replace __getitem__ entirely ---
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entity_id = self.entity_ids[idx]
        raw_label = self.labels[idx]
        timestamp = self.timestamps[idx] if self.timestamps is not None else None

        # Main row
        edf = self.db.table_dict[self.entity_table].df
        if self.entity_pkey and self.entity_pkey in edf.columns:
            erow_df = edf[edf[self.entity_pkey] == entity_id]
        else:
            erow_df = edf[edf[self.entity_col] == entity_id]
        if erow_df.empty:
            text = f"{MAIN_OPEN} Dataset: {self.dataset_name}, Table: {self.entity_table}, {self.entity_col}: {entity_id} {MAIN_CLOSE}"
            return self._format_return(text, raw_label)

        erow = erow_df.iloc[0]
        main_text  = self._row_to_text(self.entity_table, erow)
        main_block = f"{MAIN_OPEN} {main_text} {MAIN_CLOSE}"

        header = f"Dataset: {self.dataset_name}"
        current = header  # we build neighbors here; the tail is appended at the end

        # Build tail (MAIN + optional regression Q/A) and reserve budget for it
        qa_block = ""
        if self.task_type.name == "REGRESSION":
            # normalized label
            z = self._normalize_label(float(raw_label))
            # note: format to 9 decimals as requested
            qa_block = f"Question: {self._regression_question()}? Answer: <{z:.9f}>"

        # Tail is MAIN plus (for regression) the Q/A, appended as a single unit
        tail_text = main_block if qa_block == "" else (main_block + " " + qa_block)

        # 1) Collect 1-hop neighbors (strings) and the child rows actually selected
        nei1_blocks, selected_children = self._get_neighbors_1hop(erow, entity_id, timestamp)

        # 2) Add as many 1-hop blocks as fit WHILE reserving room for TAIL
        current, fit_all_1hop = self._add_blocks_with_budget(current, nei1_blocks, tail_text)

        # 3) If all 1-hop fit and 2-hop is enabled, try adding 2-hop parents (still reserving TAIL)
        if fit_all_1hop and self.include_two_hop_parents:
            nei2_blocks = self._get_parents_2hop(selected_children)
            current, _ = self._add_blocks_with_budget(current, nei2_blocks, tail_text)

        # 4) Append TAIL last
        final_text = (current + SEP + tail_text) if current else tail_text

        # Safety: if something went wrong, drop last neighbor(s) to ensure TAIL fits
        while not self._fits(final_text) and SEP in current:
            current = current.rsplit(SEP, 1)[0]
            final_text = (current + SEP + tail_text) if current else tail_text

        return self._format_return(final_text, raw_label)




    # ---------- helpers: neighbor harvesting ----------
    def _get_neighbors_1hop(
        self, entity_row: pd.Series, entity_id: int, timestamp=None
    ) -> Tuple[List[str], List[Tuple[str, pd.Series]]]:
        """
        Returns:
          nei1_blocks: list[str]  (wrapped with <1-hop NEIGHBOR> ... </...>)
          selected_children: list[(child_table, child_row)] actually used (for 2-hop)
        """
        blocks: List[str] = []
        selected_children: List[Tuple[str, pd.Series]] = []

        # Forward (entity has an FK to a parent)
        for fkey, parent_table in self.forward_edges_by_src.get(self.entity_table, []):
            if fkey in entity_row.index:
                fval = entity_row[fkey]
                if pd.notna(fval):
                    pdf = self.db.table_dict[parent_table].df
                    pkey = self.db.table_dict[parent_table].pkey_col
                    if pkey and pkey in pdf.columns:
                        for _, prow in pdf[pdf[pkey] == fval].head(1).iterrows():
                            blocks.append(self._wrap_block(self._row_to_text(parent_table, prow), hop=1))

        # Reverse (child tables referencing this entity via FK == entity_id)
        for fkey_col, child_table in self.reverse_edges_by_parent.get(self.entity_table, []):
            cdf = self.db.table_dict[child_table].df
            if fkey_col not in cdf.columns:
                continue
            rows = cdf[cdf[fkey_col] == entity_id]

            # temporal filter & sort by time desc if available
            time_col = self.db.table_dict[child_table].time_col
            if timestamp is not None and time_col and time_col in rows.columns:
                rows = rows[rows[time_col] <= timestamp]
            if time_col and time_col in rows.columns:
                rows = rows.sort_values(time_col, ascending=False)

            for _, crow in rows.head(self.per_table_k).iterrows():
                txt = self._row_to_text(child_table, crow)
                blocks.append(self._wrap_block(txt, hop=1))
                selected_children.append((child_table, crow))

        return blocks, selected_children


    def _get_parents_2hop(self, selected_children: List[Tuple[str, pd.Series]]) -> List[str]:
        """From each chosen child row, add its parent rows via f2p edges, excluding the MAIN table."""
        out: List[str] = []
        seen = set()
        for child_table, crow in selected_children:
            for fkey, parent_table in self.forward_edges_by_src.get(child_table, []):
                if parent_table == self.entity_table:  # avoid adding MAIN table again
                    continue
                if fkey not in crow.index:
                    continue
                fval = crow[fkey]
                if pd.isna(fval):
                    continue
                pdf = self.db.table_dict[parent_table].df
                pkey = self.db.table_dict[parent_table].pkey_col
                if not pkey or pkey not in pdf.columns:
                    continue
                for _, prow in pdf[pdf[pkey] == fval].head(1).iterrows():
                    key = (parent_table, pkey, prow[pkey])
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(self._wrap_block(self._row_to_text(parent_table, prow), hop=2))
        return out



    # ---------- helpers: formatting & budgeting ----------
    def _wrap_block(self, text: str, hop: int) -> str:
        if hop == 1:
            return f"{NEI1_OPEN} {text} {NEI1_CLOSE}"
        else:
            return f"{NEI2_OPEN} {text} {NEI2_CLOSE}"

    def _token_len(self, s: str) -> int:
        return len(self.tokenizer(s, add_special_tokens=True)["input_ids"])

    def _fits(self, s: str) -> bool:
        return self._token_len(s) <= self.max_length


    def _add_blocks_with_budget(self, header: str, blocks: List[str], tail_text: str) -> Tuple[str, bool]:
        """
        Greedily add blocks while reserving room for the tail_text (MAIN [+ Q/A]).
        Returns (current_text, fit_all_blocks).
        """
        current = header
        fit_all = True
        for b in blocks:
            candidate = (current + SEP + b + SEP + tail_text) if current else (b + SEP + tail_text)
            if self._fits(candidate):
                current = (current + SEP + b) if current else b
            else:
                fit_all = False
                break
        return current, fit_all
    


    def _append_blocks_with_budget(self, base_text: str, blocks: List[str]) -> str:
        """Append blocks one-by-one as long as budget allows."""
        current = base_text
        for b in blocks:
            candidate = current + SEP + b
            if self._fits(candidate):
                current = candidate
            else:
                break
        return current

    def _format_return(self, full_text: str, raw_label):
        # CLS → natural language label; REG → numeric + normalized
        if self.task_type.name in ("BINARY_CLASSIFICATION", "MULTICLASS_CLASSIFICATION"):
            if self.task_type.name == "BINARY_CLASSIFICATION":
                if self.task_name == "driver-top3":
                    q = "whether the driver is in top3"
                else:
                    q = f"whether the {self.entity_table} is positive for task '{self.task_name}'"
                ans = "yes" if int(raw_label) == 1 else "no"
                label = {"question": q, "answer": ans}
            else:
                q = f"what is the class for {self.entity_table} in task '{self.task_name}'"
                label = {"question": q, "answer": f"class_{int(raw_label)}"}
            return {"text": full_text, "label": label, "label_id": int(raw_label)}

        # Regression
        def _normalize_label(y):
            if np.isnan(y): return 0.0
            z = (y - self.label_mean) / self.label_std
            return 1e-10 if z == 0 else float(z)

        return {
            "text": full_text,
            "label": float(raw_label),
            "normalized_label": _normalize_label(float(raw_label)),
        }

    # ---------- row → text ----------
    def _row_to_text(self, table_name: str, row: pd.Series) -> str:
        """
        Convert a single table row to a compact textual block.

        Inputs
        -------
        table_name : str
            Name of the table the row comes from.
        row : pandas.Series
            One record (row) from that table.

        Output
        -------
        text : str
            A string like "Table: <name>, col1: val1, col2: val2, ...".
            - Scalars: missing checked via pd.isna; numeric columns pretty-printed as <...>.
            - Array-like with strings: preserved and joined (e.g., [a, b, c]).
            - Other array-like: summarized (first few elements).
        """
        parts = [f"Table: {table_name}"]
        numeric_cols = set(self.numeric_columns_spec.get(self.dataset_name, {}).get(table_name, []))
        time_col = self.db.table_dict[table_name].time_col

        def is_array_like(v) -> bool:
            return isinstance(v, (list, tuple, np.ndarray, pd.Series))

        def to_list(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, pd.Series):
                return v.tolist()
            return list(v)

        def array_has_str(v) -> bool:
            try:
                for el in v:
                    if isinstance(el, str):
                        return True
                return False
            except TypeError:
                return False

        for col, val in row.items():
            # Optionally skip time column
            if not self.include_temporal_context and time_col and col == time_col:
                continue

            # Array-like handling (NO pd.isna here)
            if is_array_like(val):
                seq = to_list(val)
                if array_has_str(seq):
                    # Keep strings; join them
                    val_str = "[" + ", ".join(str(s) for s in seq) + "]"
                else:
                    # Numeric/mixed arrays: summarize
                    preview = ", ".join(str(x) for x in seq[:3])
                    ell = ", …" if len(seq) > 3 else ""
                    val_str = f"[{preview}{ell}]"

            # Scalar handling (safe to use pd.isna)
            else:
                if isinstance(val, str):
                    s = val.strip()
                    if s == "" or s.lower() in {"nan", "none", "null"}:
                        val_str = "unknown"
                    else:
                        val_str = s
                elif pd.isna(val):
                    val_str = "unknown"
                elif col in numeric_cols:
                    # Pretty-print numerics in angle brackets for downstream numeric handling
                    try:
                        f = float(val)
                        val_str = f"<{int(round(f))}>" if abs(f - round(f)) < 1e-9 else f"<{f:.5f}>"
                    except Exception:
                        val_str = str(val)
                elif isinstance(val, (pd.Timestamp, np.datetime64)):
                    try:
                        val_str = pd.Timestamp(val).isoformat()
                    except Exception:
                        val_str = str(val)
                else:
                    val_str = str(val)

            parts.append(f"{col}: {val_str}")

        return ", ".join(parts)


    
    # --- add this helper anywhere in the class ---
    def _regression_question(self) -> str:
        # Task-specific phrasing; add more cases as needed.
        if self.task_name == "driver-position":
            return "what is the driver's position"
        # Fallback
        return f"what is the target value for task '{self.task_name}'"


    def _normalize_label(self, label) -> float:
        """
        Z-score normalize a scalar label using (mean, std) computed in __init__.
        Returns 0.0 for NaN/inf; replaces exact 0 with 1e-10 to avoid degenerate targets.
        """
        try:
            y = float(label)
        except Exception:
            return 0.0
        if not np.isfinite(y):
            return 0.0
        z = (y - self.label_mean) / self.label_std
        if z == 0.0:
            z = 1e-10
        return float(z)
    

    

class RelBenchTaskProcessor:
    """
    Processor to convert RelBench task data into tokenized format for LLM training.
    """
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Add special token for numbers if not present
        if "<NUM>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": ["<NUM>"]})
        
        self.num_token = "<NUM>"
        self.num_id = tokenizer(self.num_token, add_special_tokens=False)["input_ids"][0]
        self.NUM_RE = re.compile(
            r"<([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)>"
        )
    
    def process_dataset(
        self,
        dataset: RelBenchTaskDataset,
        save_path: str = None,
        chunk_size: int = 10000,
    ) -> List[Dict[str, torch.Tensor]]:
        """Process the entire dataset and optionally save it."""
        
        processed_samples = []
        
        for idx in tqdm(range(len(dataset)), desc=f"Processing {dataset.split} samples"):
            sample = dataset[idx]
            processed = self._process_sample(sample, dataset.task_type.name)
            processed_samples.append(processed)
            
            # Save chunks periodically
            if save_path and (idx + 1) % chunk_size == 0:
                chunk_path = save_path.replace(".pt", f"_chunk_{(idx + 1) // chunk_size}.pt")
                torch.save(processed_samples[-chunk_size:], chunk_path)
                print(f"Saved chunk to {chunk_path}", file=sys.stderr, flush=True)
        
        if save_path:
            torch.save(processed_samples, save_path)
            print(f"Saved {len(processed_samples)} processed samples to {save_path}", file=sys.stderr, flush=True)
        
        return processed_samples
    
    def _process_sample(self, sample: Dict, task_type: str) -> Dict[str, torch.Tensor]:
        """Process a single sample for training."""
        
        text = sample["text"]
        label = sample["label"]
        normalized_label = sample["normalized_label"]
        
        # Replace numbers and capture values
        proc_text, numbers = self._replace_nums(text)
        
        # Create prompt based on task type
        if task_type == "BINARY_CLASSIFICATION":
            label_text = "Yes" if label == 1 else "No"
            full_text = f"{proc_text} Question: Will the event occur? Answer: {label_text}"
        elif task_type == "MULTICLASS_CLASSIFICATION":
            full_text = f"{proc_text} Question: What is the class? Answer: Class_{int(label)}"
        elif task_type == "REGRESSION":
            # For regression, add <NUM> token for the target
            full_text = f"{proc_text} Question: What is the value? Answer: <NUM>"
            numbers.append(str(normalized_label))
        else:
            full_text = proc_text
        
        # Tokenize
        enc = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        ids = enc.input_ids[0]
        attn = enc.attention_mask[0]
        L = ids.size(0)
        
        # Build shifted labels for language modeling
        labels = ids.clone()
        labels[:-1] = ids[1:]
        labels[-1] = -100
        
        # Prepare numeric embedding and regression targets
        num_embed = torch.zeros(L)
        num_target = torch.zeros(L)
        num_mask = torch.zeros(L)
        
        # Inject numeric values where input_ids == <NUM>
        num_positions = (ids == self.num_id).nonzero(as_tuple=True)[0]
        for i, pos in enumerate(num_positions):
            if i < len(numbers):
                try:
                    num_embed[pos] = float(numbers[i])
                except:
                    pass
        
        # Mark regression targets where labels == <NUM>
        target_positions = (labels == self.num_id).nonzero(as_tuple=True)[0]
        for i, pos in enumerate(target_positions):
            if i < len(numbers):
                try:
                    num_target[pos] = float(numbers[i])
                    num_mask[pos] = 1.0
                except:
                    pass
        
        # Mask everything before "Answer:" to only supervise the answer
        answer_start = self._find_answer_start(full_text)
        if answer_start > 0:
            # Find token position corresponding to answer start
            answer_tokens = self.tokenizer(full_text[:answer_start], add_special_tokens=True)["input_ids"]
            answer_token_pos = len(answer_tokens)
            
            labels[:answer_token_pos] = -100
            num_target[:answer_token_pos] = 0.0
            num_mask[:answer_token_pos] = 0.0
        
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": labels,
            "num_embed": num_embed,
            "num_target": num_target,
            "num_mask": num_mask,
            "original_label": torch.tensor(label, dtype=torch.float32),
            "normalized_label": torch.tensor(normalized_label, dtype=torch.float32),
        }
    
    def _replace_nums(self, text: str) -> Tuple[str, List[str]]:
        """Replace <number> with <NUM> and return (processed_text, numbers_list)."""
        numbers = []
        
        def repl(m):
            numbers.append(m.group(1))
            return self.num_token
        
        processed = self.NUM_RE.sub(repl, text)
        return processed, numbers
    
    def _find_answer_start(self, text: str) -> int:
        """Find the character position where 'Answer:' starts."""
        answer_idx = text.find("Answer:")
        return answer_idx if answer_idx >= 0 else -1



def create_dataloader(
    processed_samples: List[Dict[str, torch.Tensor]],
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader with proper padding."""
    
    def collate_fn(batch):
        keys = batch[0].keys()
        max_len = max(len(b["input_ids"]) for b in batch)
        
        collated = {}
        for key in keys:
            if key in ["input_ids", "attention_mask", "labels", "num_embed", "num_target", "num_mask"]:
                # Pad sequences
                pad_val = -100 if key == "labels" else 0
                padded = []
                for b in batch:
                    tensor = b[key]
                    if len(tensor) < max_len:
                        padding = torch.full((max_len - len(tensor),), pad_val, dtype=tensor.dtype)
                        tensor = torch.cat([tensor, padding])
                    padded.append(tensor)
                collated[key] = torch.stack(padded)
            else:
                # Stack other tensors directly
                collated[key] = torch.stack([b[key] for b in batch])
        
        return collated
    
    return DataLoader(
        processed_samples,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build neighbor-aware texts and tokenize them for RelBench tasks.")
    parser.add_argument("--dataset", type=str, default='rel-amazon', help="RelBench dataset name, e.g., rel-stack")
    parser.add_argument("--task", type=str, default='item-churn', help="RelBench task name, e.g., user-badge")
    parser.add_argument("--per_table_k", type=int, default=3, help="Max neighbors per child table (1-hop)")
    parser.add_argument("--two_hop", action="store_false",default=True, help="Enable adding 2-hop parents if budget permits")
    parser.add_argument("--max_length", type=int, default=1536, help="Token budget for constructing a sample")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for saving processed tensors")
    args = parser.parse_args()


    # --- keep these lines as requested ---
    model_path = "/sunmingyang/xiaomin/lxr/TabularFM/model/qwen_tokenizer/" #! absolute path
    model_path = "./model/qwen_tokenizer/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with open("data/float_encoder_columns.json", "r") as f:
        numeric_columns_spec = json.load(f)
    save_dir = "/sunmingyang/xiaomin/lxr/TabularFM/data/precomputed_2nd_stage/text_input/" #! absolute path
    save_dir = "./data/precomputed_2nd_stage/text_input/"
    # -------------------------------------


    # Replace your current per-split block with this JSONL appender.
    # It appends every 1,000 samples to ONE file per split and drops the final remainder.

    splits = ["train","val", "test"]
    for split in splits:
        print (f"current: {split}")
        # 1) Build textual samples (neighbors + MAIN; for regression, Q/A already appended)
        ds = RelBenchTaskDataset(
            dataset_name=args.dataset,
            task_name=args.task,
            split=split,
            numeric_columns_spec=numeric_columns_spec,
            per_table_k=args.per_table_k,
            include_two_hop_parents=args.two_hop,
            tokenizer=tokenizer,
            max_length=args.max_length if split=='train' else 5092,
            download=False,
        )

        out_dir = os.path.join(save_dir, args.dataset, args.task, split)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "raw_input.jsonl")  # single file per split

        # ----- resume: count existing lines to decide start index -----
        start_idx = 0
        if os.path.exists(out_path):
            # Count valid lines (best-effort; corrupt lines won't crash resume)
            with open(out_path, "r", encoding="utf-8") as f:
                line_count = 0
                for _ in f:
                    line_count += 1
            start_idx = line_count

        total = len(ds)
        if start_idx >= total:
            print(f"[{args.dataset}/{args.task}/{split}] already complete "
                  f"({start_idx}/{total} lines in {out_path}). Skipping.", file=sys.stderr, flush=True)
            continue

        print(f"[{args.dataset}/{args.task}/{split}] resuming at index {start_idx} "
              f"→ will append into {out_path} (total={total}).", file=sys.stderr, flush=True)

        buf = []
        saved = start_idx  # already on disk
        skipped = 0

        # Iterate from where we left off
        for i in tqdm(range(start_idx, total), desc=f"Processing {split}"):
            try:
                ex = ds[i]
                text = ex["text"]

                # Append Q/A tail for classification tasks (labels returned as dict)
                lbl = ex.get("label", None)
                if isinstance(lbl, dict):
                    q = (lbl.get("question", "") or "").rstrip("?")
                    a = (lbl.get("answer", "") or "")
                    text = f"{text} Question: {q}? Answer: {a}"

                buf.append({"text": text})
                # Append chunk to file
                if len(buf) >= args.chunk_size:
                    with open(out_path, "a", encoding="utf-8") as f:
                        for obj in buf:
                            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        print (f"saving to {out_path}", file=sys.stderr, flush=True)
                    saved += len(buf)
                    buf.clear()
            except Exception:
                skipped += 1
                print (f"skip, exception", file=sys.stderr, flush=True)
                continue

        # Flush any remainder (so the split can be fully complete on success)
        if buf:
            with open(out_path, "a", encoding="utf-8") as f:
                for obj in buf:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            saved += len(buf)
            buf.clear()

        print(
            f"[{args.dataset}/{args.task}/{split}] now at {saved}/{total} lines in {out_path}; "
            f"skipped {skipped} samples.", file=sys.stderr, flush=True
        )
        
        
        
        