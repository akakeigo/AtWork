"""
numeric_llama.py
────────────────
Fine‑tune a LLaMA model so that the special token <NUM> is predicted
with (1) cross‑entropy in the token stream  and  (2) an MSE regression
head that outputs the actual float value.

Requires:
  pip install torch datasets transformers==4.41 openai tqdm
"""
import os
os.environ["TRANSFORMERS_NO_META_DEVICE"] = "1"
import sys
import re
from typing import List, Dict, Any, Optional
from termcolor import colored
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModelForCausalLM
)
from tqdm.auto import tqdm

from floatEncoder import SignedFloatAutoencoder
from PretrainingDataset import RelBenchRowDataset
import warnings
warnings.filterwarnings("ignore")

# helper

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

    # --- 1) Cross‐Entropy per token & per sample ---
    # flatten for F.cross_entropy
    flat_logits = logits.view(-1, V)               # (B*L, V)
    flat_labels = labels.view(-1)                  # (B*L,)
    ce_tok = F.cross_entropy(
        flat_logits,
        flat_labels,
        reduction="none",
        ignore_index=-100
    ).view(B, L)                                   # (B, L)

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

    # ---------- public helpers --------------------------------------
    def make_dataset(self, texts: List[str]) -> NumericDataset:
        """Return torch Dataset of tokenised training examples."""
        samples = [self._encode_one(t) for t in texts]
        return NumericDataset(samples)

    def make_dataloader(
        self,
        texts: List[str],
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 4,
    ) -> DataLoader:
        ds = self.make_dataset(texts)

        def collate(batch):
            # simple pad‑to‑longest
            keys = batch[0].keys()
            max_len = max(len(b["input_ids"]) for b in batch)
            collated = {}
            for k in keys:
                if k in ("num_mask",):
                    pad_val = 0
                else:
                    pad_val = 0
                collated[k] = torch.stack(
                    [
                        torch.nn.functional.pad(
                            b[k], (0, max_len - len(b[k])), value=pad_val
                        )
                        for b in batch
                    ],
                    dim=0,
                )
            return collated

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
        )


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
    def __init__(
        self,
        base_name: str = "Qwen/Qwen2-1.5B",
        float_encoder = None,
    ):
        super().__init__()
        self.qwen = AutoModelForCausalLM.from_pretrained(
            base_name,
            trust_remote_code=True,
            output_hidden_states=True,
            device_map={"": "cuda:0"},    # put entire model on cuda:0
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,       # <— turn this OFF!
        )
        
        d_model = self.qwen.config.hidden_size
        # numeric regression head (same as before)
        self.num_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        head_dtype = next(self.qwen.parameters()).dtype  # torch.bfloat16
        self.num_head = self.num_head.to(head_dtype)
        # plug‑in float encoder
        self.float_encoder = float_encoder
        # self.float_encoder = nn.Linear(1,1536)

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
    """
    Simple single‑GPU trainer loop for demonstration.
    """

    def __init__(
        self,
        model: QwenNumericModel,
        tokenizer,
        lr: float = 1e-5,
        epochs: int = 1,
    ):
        self.model = model
        self.tok = tokenizer
        self.lr = lr
        self.epochs = epochs
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def fit_epoch(self, dataloader: DataLoader):
        """
        Run a single training epoch over `dataloader`.

        Args:
            dataloader (DataLoader): Yields dicts with keys
                input_ids, attention_mask, labels,
                num_embed, num_target, num_mask.

        Returns:
            avg_loss (float):   average total loss (CE + MSE),
            avg_ce   (float):   average cross-entropy loss,
            avg_mse  (float):   average numeric regression loss.
        """
        
        self.model.train()

        total_loss = 0.0
        total_ce   = 0.0
        total_mse  = 0.0
        n_batches  = 0

        pbar = tqdm(dataloader, desc="Training epoch", leave=False)
        for batch in pbar:
            # 1) move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            # for k in batch:
            #     if k=='labels':
            #         print ('labelId')
            #         print (batch[k])
            # print (batch)
            # 2) forward + backward
            total_ps, ce_ps, mse_ps = self.model(**batch)
            valid = ~torch.isnan(total_ps)
            # sys.exit(0)
            # keep only valid ones
            if valid.any():
                loss = total_ps[valid].mean()
                ce_loss = ce_ps[valid].mean()
                mse_loss = mse_ps[valid].mean()
                loss.backward()
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                total_loss += loss.item()
                total_ce   += ce_loss.item()
                total_mse  += mse_loss.item()
                n_batches  += 1
            else:
                # all NaN? skip this batch or handle specially
                print("Warning: all per-sample losses NaN, skipping batch")
                continue

            pbar.set_postfix(
                loss=total_loss / n_batches,
                ce=  total_ce   / n_batches,
                mse= total_mse  / n_batches,
            )

        # 4) final averages
        avg_loss = total_loss / max(1, n_batches)
        avg_ce   = total_ce   / max(1, n_batches)
        avg_mse  = total_mse  / max(1, n_batches)
        return avg_loss, avg_ce, avg_mse

    # @torch.no_grad()
    # def infer(self, prompt: str, max_new_tokens: int = 30):
    #     self.model.eval()
    #     out_txt, nums = self.model.generate_with_numbers(
    #         prompt, self.tok, max_new_tokens=max_new_tokens
    #     )
    #     return out_txt, nums


# ---------------------------------------------------------------------
# 4)  Usage example
# ---------------------------------------------------------------------
if __name__ == "__main__":
    device = 'cuda:0'
    model_name = "Qwen/Qwen2-1.5B"          # or any 1 B–class checkpoint
    tok   = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.add_special_tokens({"additional_special_tokens": ["<NUM>"]})


    encoder_path = f"model/float_encoder.pt"
    float_encoder = SignedFloatAutoencoder(d_model=1536, d_mant=1536/2, emin=-20, emax=20, 
                 predict_log=True, lambda_sign=1.0, base=2)
    float_encoder.load_state_dict(torch.load(encoder_path))
    float_encoder.to(device)

    # ----- 1. prepare tiny toy data ------
    # texts = [
    #     "Dataset: rel-avito, Table: VisitStream, IPID: 398015, UserID: 86464, AdID: 2319841, ViewDate: 2015-05-08 07:05:23",
    #     "Dataset: rel-amazon, Table: product, category: ['Books' 'Crafts, Hobbies &amp; Home' 'Pets &amp; Animal Care'], product_id: 174428, brand: Phillip Purser, title: Corn &amp; Rat Snakes (Complete Herp Care), description: PHILIP PURSER is an avid herp and fish hobbyist, having authored many articles and books on these and other topics. He lives in Temple, Georgia., price: <-0.12062>."
    # ]
    
    datasets = [
        # "rel-amazon",
        "rel-avito",
        # "rel-event",
        # "rel-f1",
        # "rel-hm",
        # "rel-stack",
        # "rel-trial",
    ]

    json_path = "data/float_encoder_columns.json"
    
    ds = RelBenchRowDataset(datasets, json_path, download=True)
    
    processor = NumericDataProcessor(tok, max_length=512)
    # ----- 2. build model ---------------
    model = QwenNumericModel(base_name="Qwen/Qwen2-1.5B",float_encoder=float_encoder)
    model.to(device)
    model.qwen.resize_token_embeddings(len(tok))  # reflect <NUM>

    # ----- 3. train ----------------------
    trainer = QwenNumericTrainer(model, tok, epochs=1, lr=5e-6)
    batch_size = 10
    epochs = 1
    for e in range(epochs):
        print ('epoch 1')
        tot_samples = []
        loader = DataLoader(
            ds,
            batch_size=5,
            shuffle=True,           # randomly sample each epoch
            num_workers=4,          # adjust to how many CPU cores you want to use
            collate_fn=collate_to_list,
            drop_last=False,        # or True if you want to drop the tail
        )
        
        print ('loading samples')
        cnt = 0
        for l in tqdm(loader):
            tot_samples.extend(l)
            cnt += 1
            if cnt>10: break
        print (colored('make dataset','red'))
        dloader = processor.make_dataloader(tot_samples, batch_size=batch_size)
        print ('training')
        trainer.fit_epoch(dloader)

    # ----- 4. inference ------------------
    # prompt = "The price of item 3 is"
    # gen, values = trainer.infer(prompt, max_new_tokens=10)
    # print("Generated:", gen)
    # print("Numeric values:", values)
