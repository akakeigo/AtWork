import sys
import json, random
from pathlib import Path
from typing import List, Dict, Tuple, Any

import pandas as pd
from torch.utils.data import Dataset
from relbench.datasets import get_dataset
import numpy as np
import math

def _fmt_num_in_angle(x: str) -> str:
    # x already a string from _clean_value; convert safely
    try:
        f = float(x)
    except Exception:
        return f"<{x}>"
    return f"<{f:.5f}>"

def _fmt_general(x: str) -> str:
    """
    If x is a float string like '398015.0' or '4.00', drop the decimal part.
    Otherwise return as-is.
    """
    try:
        f = float(x)
        # check if it's effectively an integer
        if math.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        # otherwise keep original (do not force 5 decimals here)
        return x
    except Exception:
        return x


class RelBenchRowDataset(Dataset):
    """
    Sample‑level dataset that, on every __getitem__, returns *one* randomly
    chosen row from the RelBench corpus formatted as a plain‑text string.

    Format:
        "Dataset: <ds>, Table: <tbl>, col1: val1, col2: val2, ..."

    Numeric columns listed in the JSON spec are *not* yet wrapped in <NUM>;
    that can be done upstream when converting the text to LLM input.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
class RelBenchRowDataset(Dataset):
    def __init__(
        self,
        datasets: List[str],
        json_path: str,
        download: bool = True,
        seed: int = 1,
    ):
        super().__init__()
        self._rng = random.Random(seed)

        # 1) numeric-column spec
        with open(json_path, "r") as f:
            self.num_spec: Dict[str, Dict[str, List[str]]] = json.load(f)

        # 2) load & normalize
        self.tables: Dict[str, Dict[str, pd.DataFrame]] = {}
        for ds in datasets:
            ds_obj = get_dataset(name=ds, download=download)
            db = ds_obj.get_db()

            tbl_dict: Dict[str, pd.DataFrame] = {}
            for tbl_name, tbl_obj in db.table_dict.items():
                df = tbl_obj.df.copy()
                cols_to_norm = self.num_spec.get(ds, {}).get(tbl_name, [])
                for col in cols_to_norm:
                    if col not in df.columns:
                        continue

                    # dropna to compute stats
                    column_values = pd.to_numeric(df[col], errors="coerce").dropna()
                    if column_values.empty:
                        # nothing to normalize
                        continue

                    mean = column_values.mean()
                    std  = column_values.std()

                    # protect against std == 0 or NaN
                    if not std or pd.isna(std):
                        df[col] = 0.0
                    else:
                        # element-wise (x - mean)/std with 0→1e-10 replacement
                        def _norm_scalar(v):
                            try:
                                x = float(v)
                            except (ValueError, TypeError):
                                return np.nan
                            z = (x - mean) / std
                            if z == 0:
                                z = 1e-10
                            return z

                        df[col] = pd.to_numeric(df[col], errors="coerce").map(_norm_scalar)

                tbl_dict[tbl_name] = df

            self.tables[ds] = tbl_dict

        self.total_rows = sum(len(df) for tmap in self.tables.values() for df in tmap.values())

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Return total number of rows across all loaded tables."""
        return self.total_rows

    def __getitem__(self, idx: int) -> str:
        rng = self._rng
        ds_name  = rng.choice(list(self.tables.keys()))
        tbl_map  = self.tables[ds_name]
        tbl_name = rng.choice(list(tbl_map.keys()))
        df       = tbl_map[tbl_name]

        row = df.iloc[rng.randrange(len(df))]
        num_cols = set(self.num_spec.get(ds_name, {}).get(tbl_name, []))

        parts = []
        for col, raw_val in row.items():
            val = self._clean_value(raw_val)

            if col in num_cols and val != "unknown":
                # numeric column → keep 5 decimals in <...>
                val_str = _fmt_num_in_angle(val)
            else:
                # non-numeric or unknown → if looks like int-ish float, strip .0
                val_str = _fmt_general(val)

            parts.append(f"{col}: {val_str}")

        rng.shuffle(parts)
        return ", ".join([f"Dataset: {ds_name}, Table: {tbl_name}"] + parts)


    @staticmethod
    def _clean_value(val) -> str:
        """
        Convert NaN / 'nan' / empty‑like values to the literal 'unknown'.
        Otherwise return str(val).
        """
        if isinstance(val, str):
            if val.strip() == "" or val.lower() == "nan" or 'nan' in val.lower() or 'none' in val.lower():
                return "unknown"
        return str(val)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def get_numeric_columns(self, ds: str, tbl: str) -> List[str]:
        """Return list of numeric columns for (dataset, table) per JSON spec."""
        return self.num_spec.get(ds, {}).get(tbl, [])


# ─────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    datasets = [
        "rel-amazon",
        "rel-avito",
        "rel-f1",
        "rel-hm",
        "rel-stack",
        "rel-trial",
    ]

    json_path = "data/float_encoder_columns.json"
    
    ds = RelBenchRowDataset(datasets, json_path, download=True)
    print("Total rows:", len(ds))
    
    samples = [ds[i] for i in range(10000)]

    # Step 2: filter those containing exact "Table: studies"
    filtered = [s for s in samples if "Table: studies" in s]

    print(f"Total samples with 'Table: studies': {len(filtered)}")

    # Step 3: randomly pick 3 and print
    if filtered:
        chosen = random.sample(filtered, k=min(3, len(filtered)))
        for i, s in enumerate(chosen, 1):
            print(f"\nSample {i}:\n{s}")
    else:
        print("No samples with 'Table: studies' found.")
    # print (ds[0])
    # sys.exit(0)
    # Fetch 3 random samples
    # for v in range(300):
    #     if "Table: studies" in ds[v]:
    #         print ()
    #         print(ds[v])  # idx ignored
    #         print ("Table: studies" in ds[v])
    #         print ()
