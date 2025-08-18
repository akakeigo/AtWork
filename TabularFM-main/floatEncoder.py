import sys
import random, itertools
import math
import torch
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Robust patch for torch.empty
#   • Remove None‑valued dtype / device
#   • Cast the size tuple to ints, because some callers pass floats
# ─────────────────────────────────────────────────────────────────────
_orig_empty = torch.empty
def _empty_no_none(shape, *args, **kwargs):
    # 1) strip None kwargs
    if kwargs.get("dtype") is None:
        kwargs.pop("dtype", None)
    if kwargs.get("device") is None:
        kwargs.pop("device", None)

    # 2) ensure shape is a tuple of *ints*
    if isinstance(shape, tuple):
        shape = tuple(int(s) for s in shape)
    else:
        shape = (int(shape),)

    return _orig_empty(shape, *args, **kwargs)

torch.empty = _empty_no_none

import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
from torch.utils.data import Dataset, DataLoader
from relbench.datasets import get_dataset
from termcolor import colored
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------
# Extract numeric values from a RelBench table
# --------------------------------------------------------------------
def extract_numeric_values(table, columns: list[str]) -> list[float]:
    """
    Pulls out all valid floats from the specified columns of a table, normalizing them.
    
    Args:
        table: RelBench table object (has .df DataFrame)
        columns: list of column names to scan
    
    Returns:
        list[float]: all numeric entries found after normalization
    """
    vals = []
    df = table.df
    
    for col in tqdm(columns):
        
        if col in df.columns:
            # Drop NaN values from the column
            column_values = df[col].dropna()

            # Calculate mean and std
            mean = column_values.mean()
            std = column_values.std()

            # Normalize the column values (x - mean) / std
            for v in column_values:
                try:
                    normalized_value = (v - mean) / std
                    vals.append(normalized_value)
                except (ValueError, TypeError):
                    continue
    return vals


# Helper: decompose float to sign, exponent, mantissa
def decompose(x: torch.Tensor, base: int = 2):
    sign = (x < 0).long()
    abs_x = x.abs().clamp_min(1e-30)
    exponent = torch.floor(torch.log(abs_x) / math.log(base)).long()
    mantissa = abs_x / (base ** exponent.float())
    return sign, exponent, mantissa


class SignedFloatAutoencoder(nn.Module):
    """
    Float autoencoder with joint log-magnitude (MSE) and sign (BCE) prediction.
    """
    def __init__(self, d_model=32, d_mant=16, emin=-20, emax=20, 
                 predict_log=True, lambda_sign=1.0, base=2):
        super().__init__()
        self.predict_log = predict_log
        self.lambda_sign = lambda_sign
        self.base = base
        # Encoder: sign, exponent, mantissa embeddings
        # Sign embedding
        self.sign_emb = nn.Embedding(2, d_model // 4)
        # Exponent embedding
        self.exp_emb = nn.Embedding(emax - emin + 1, d_model // 4)
        # Mantissa MLP
        self.mant_mlp = nn.Sequential(
            nn.Linear(1, d_mant), nn.GELU(), nn.Linear(d_mant, d_mant)
        )
        # Final projection
        concat_dim = (d_model // 4) * 2 + d_mant
        self.proj = nn.Linear(concat_dim, d_model)
        # Decoder heads
        self.mag_head = nn.Linear(d_model, 1)   # predict log-mag
        self.sign_head = nn.Linear(d_model, 1)  # predict sign logit

        self.emin, self.emax = emin, emax

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        sign, exp, mant = decompose(x, base=self.base)
        sign_vec = self.sign_emb(sign)
        exp_idx = (exp.clamp(self.emin, self.emax) - self.emin).long()
        exp_vec = self.exp_emb(exp_idx)
        mant_vec = self.mant_mlp(mant.unsqueeze(-1))
        z = torch.cat([sign_vec, exp_vec, mant_vec], dim=-1)
        return self.proj(z)

    def decode(self, h: torch.Tensor):
        """
        Returns:
            log_mag_pred: Tensor[...,] predicted log(|x|)
            sign_logit:  Tensor[...,] predicted sign logit
        """
        log_mag_pred = self.mag_head(h).squeeze(-1)
        sign_logit = self.sign_head(h).squeeze(-1)
        return log_mag_pred, sign_logit

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        log_mag, sign_logit = self.decode(h)
        # Convert log-mag back to x_hat if needed
        if self.predict_log:
            x_hat = torch.exp(log_mag)
        else:
            x_hat = log_mag
        # Apply sign
        return x_hat * torch.sign(x), log_mag, sign_logit

    def step(self, x: torch.Tensor, optimizer: torch.optim.Optimizer):
        optimizer.zero_grad()
        h = self.encode(x)
        log_mag_pred, sign_logit = self.decode(h)
        # Magnitude loss (MSE on log |x|)
        target_log = torch.log(x.abs().clamp_min(1e-45))
        loss_mag = F.mse_loss(log_mag_pred, target_log)
        # Sign loss (BCE on sign)
        sign_true = (x >= 0).float()
        loss_sign = F.binary_cross_entropy_with_logits(sign_logit, sign_true)
        loss = loss_mag + self.lambda_sign * loss_sign
        loss.backward()
        optimizer.step()
        return loss.item()


    def fit(
        self,
        data: Sequence[float],
        epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: bool = True
    ):
        """
        Train the autoencoder, using 75% of data for training and 25% for validation.
        After each epoch, evaluate on the held-out split.
        """
        device = next(self.parameters()).device

        # 1) Convert to tensor and shuffle
        tensor = torch.tensor(data, dtype=torch.float32, device=device)
        perm0 = torch.randperm(len(tensor), device=device)
        
        # 2) Split 75% train / 25% val
        split_idx = int(0.75 * len(tensor))
        train_tensor = tensor[perm0[:split_idx]]
        val_list     = tensor[perm0[split_idx:]].cpu().tolist()  # for evaluate()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # 3) Epoch loop
        for ep in range(1, epochs + 1):
            # shuffle training data each epoch
            perm = torch.randperm(len(train_tensor), device=device)
            losses = []

            # mini‑batch updates
            for i in tqdm(range(0, len(train_tensor), batch_size)):
                batch = train_tensor[perm[i : i + batch_size]]
                losses.append(self.step(batch, optimizer))

            # 4) Evaluate on validation split
            val_metrics = self.evaluate(val_list)

            if verbose:
                train_loss = sum(losses) / len(losses)
                print(
                    f"Epoch {ep:02d} | "
                    f"train-loss {train_loss:.4e} | "
                    f"val-mse-log {val_metrics['mse_log']:.4e} | "
                    f"val-bce-sign {val_metrics['bce_sign']:.4e}"
                )


    def evaluate(
        self,
        data: Sequence[float],
        batch_size: int = 1024
    ):
        self.eval()
        device = next(self.parameters()).device

        total_mag_loss = 0.0
        total_sign_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                batch_vals = data[start : start + batch_size]
                x = torch.tensor(batch_vals, dtype=torch.float32, device=device)
                B = x.size(0)

                # forward
                h = self.encode(x)                      # (B, ...)
                log_mag_pred, sign_logit = self.decode(h)  # (B,), (B,)

                # compute targets
                target_log = torch.log(x.abs().clamp_min(1e-45))  # (B,)
                sign_true = (x >= 0).float()                      # (B,)

                # sum‑reduction losses
                mag_loss = F.mse_loss(log_mag_pred, target_log, reduction='sum').item()
                sign_loss = F.binary_cross_entropy_with_logits(
                    sign_logit, sign_true, reduction='sum'
                ).item()

                total_mag_loss += mag_loss
                total_sign_loss += sign_loss
                total_count += B

        # average over all samples
        avg_mag_loss = total_mag_loss / total_count
        avg_sign_loss = total_sign_loss / total_count

        self.train()
        return {
            "mse_log": avg_mag_loss,
            "bce_sign": avg_sign_loss,
        }


# --------------------------------------------------------------------
# Main pipeline: train encoders for each dataset/table
# --------------------------------------------------------------------

def augment(values: np.ndarray) -> np.ndarray:
    """
    Augment a numpy array of floats by:
      1. Inserting the range [-5000, …, 5000] with step 1.
      2. For each element x in the combined array, generating 5 noisy copies:
         - Noise drawn from N(0, σ²), where σ is randomly chosen from [1.0, 0.1, 5.0, 10.0].
      3. Replacing any exact zero in the result with 1e-5.
      4. Shuffling the final array.

    Args:
        values (np.ndarray): 1D array of float values.

    Returns:
        np.ndarray: 1D array of augmented floats.
    """
    
    rng = np.random.default_rng(42)

    # 1. Insert the full range -5000 to 5000
    edge = np.arange(-5000, 5001, dtype=np.float32)
    arr = np.concatenate((values.astype(np.float32), edge))

    # 2. Repeat each element 5× and add Gaussian noise
    k = 15
    arr_rep = np.repeat(arr, k)  # shape = (arr.size * k,)

    # Choice of standard deviations
    std_choices = np.array([1.0, 0.1, 5.0, 10.0], dtype=np.float32)
    # For each repeated element, pick a σ at random
    stds = rng.choice(std_choices, size=arr_rep.shape)
    # Sample noise ~ N(0, σ²)
    noise = rng.standard_normal(size=arr_rep.shape) * stds

    augmented = arr_rep + noise

    # 3. Replace zeros (if any) with a small constant
    augmented[augmented == 0] = 1e-5

    # 4. Shuffle the result
    rng.shuffle(augmented)

    return augmented



def train_float_encoder(
    config_path: str,
    device: str = "cuda:0",
    d_model: int = 32,
    d_mant: int = 16,
    emin: int = -20,
    emax: int = 20,
    lambda_sign: float = 1.0,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    
    """Aggregate numeric values across **all** datasets / tables defined in
    `config_path` and train a *single* `SignedFloatAutoencoder` on the full pool.

    Args:
        config_path: Path to JSON mapping  ⇢  {dataset: {table: [cols]}}.
        device: "cuda" or "cpu".
        d_model, d_mant, emin, emax, lambda_sign: model hyper‑parameters.
        epochs, batch_size, lr: optimisation hyper‑parameters.

    Returns:
        A trained `SignedFloatAutoencoder` covering *all* numeric columns.
    """
    # 1 ▸ load schema
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 2 ▸ collect floats from every dataset / table / column
    all_floats: list[float] = []
    cnt = 0
    for ds_name, table_map in config.items():
        if ds_name == 'rel-amazon': continue
        cnt +=1
        # if cnt>2: break
        print(colored(f"\n=== Scanning dataset: {ds_name} ===", "red"))
        ds = get_dataset(name=ds_name, download=True)
        db = ds.get_db()

        for table_name, cols in table_map.items():
            if not cols:
                continue  # skip tables with no numeric cols in schema
            print (table_name, cols)
            table = db.table_dict.get(table_name)
            if table is None:
                continue  # table absent

            vals = extract_numeric_values(table, cols)
            print (colored(f"table len: {len(vals)}", "yellow"))
            if len(vals)>0:
                print(colored(f"  • Table '{table_name}' columns={cols}, max_val:{np.asarray(vals).max()}, min_val:{np.asarray(vals).min()}, mean:{np.asarray(vals).mean()}",'red'))
            all_floats.extend(vals)
    # Augment the data
    all_floats = list(set(all_floats))
    # Convert list to numpy array for faster processing
    all_floats_np = np.array(all_floats)
    print (f"{len(all_floats)}, {all_floats_np.min()}, {all_floats_np.max()}, {all_floats_np.mean()}")
    # save numpy array to file
    np.save('data/float_values.npy', all_floats_np)
    # all_floats_np = np.load('data/float_values.npy')
    all_floats = augment(all_floats_np)
    print (len(all_floats), all_floats.min(), all_floats.max(), all_floats.mean())
    # if not all_floats:
    #     raise ValueError("No numeric values found across all datasets.")

    # 3 ▸ instantiate encoder
    encoder = SignedFloatAutoencoder(
        d_model=d_model,
        d_mant=d_mant,
        emin=emin,
        emax=emax,
        lambda_sign=lambda_sign,
    ).to(device)

    # 4 ▸ train
    encoder.fit(
        data=all_floats,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=True,
    )
    return encoder


if __name__ == "__main__":
    JSON_PATH = "data/float_encoder_columns.json"
    encoder = train_float_encoder(
        config_path=JSON_PATH,
        device='cuda:0',
        d_model=896,
        d_mant=896/2,
        emin=-20,
        emax=20,
        lambda_sign=1.0,
        epochs=20,
        batch_size=5024,
        lr=1e-3
    )
    
    save_path = "model/float_encoder.pt"
    torch.save(encoder.state_dict(), save_path)
    print(f"Encoder weights saved to: {save_path}")
    