# Before running, export your OpenAI API key in your shell:
#   export OPENAI_API_KEY="your_api_key_here"

import os

# Read the API key from the environment (requires openai >= 1.20)
import openai
import sys
from termcolor import colored
from openai import OpenAI
import json

client = OpenAI()

def detect_float_encoder_columns(dataset_names):
    """
    Analyze RelBench tables to determine which columns require float encoding.

    Inputs:
    - dataset_names: List[str] of RelBench dataset identifiers.
    
    Returns:
    - float_encoder_dict: Dict[str, Dict[str, List[str]]]
      Mapping dataset_name -> { table_name: [columns_to_float_encode] }
    """
    from relbench.datasets import get_dataset

    float_encoder_dict = {}
    
    for ds_name in dataset_names:
        print (colored(f"Processing dataset: {ds_name}", "red"))
        ds = get_dataset(name=ds_name, download=True)
        db = ds.get_db()
        float_encoder_dict[ds_name] = {}
        
        for table_name, table in db.table_dict.items():
            sample = table.df.iloc[0:20].to_string(index=False)
            
            prompt = f"""Given the sample rows for table '{table_name}' from dataset '{ds_name}':
{sample}

Identify which column names require float encoding (numeric or ordinal features where proximity matters),
excluding ID-like, and purely categorical identifier columns, and exclude date and time-related columns, and the columns that can be easily transformed into categorical texts, for instance, a column: interested with 0 and 1, this can be transformed into a categorical column. Think step by step and finally output the column names in the format:
<Columns>
col1, col2, ...

If none, respond:
<Columns>
"""
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": (
                        "Output the <Columns> section as specified."
                    )},
                    {"role": "user", "content": prompt}
                ]
            )
            text = resp.choices[0].message.content.strip()
            print (colored(text, "yellow"))
            # Parse columns
            
            cols = []
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if line.strip() == "<Columns>":
                    if i+1 < len(lines) and lines[i+1].strip():
                        cols = [c.strip() for c in lines[i+1].split(",")]
                    break
            float_encoder_dict[ds_name][table_name] = cols
            print (colored(f"float cols:{cols}", "green"))

    return float_encoder_dict



datasets = ['rel-amazon',
 'rel-avito',
 'rel-event',
 'rel-f1',
 'rel-hm',
 'rel-stack',
 'rel-trial']


save_path = './data/float_encoder_columns.json'
results = detect_float_encoder_columns(datasets)
print(results)


# Save results to JSON
with open(save_path, "w") as fp:
    json.dump(results, fp, indent=2)
