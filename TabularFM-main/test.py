# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)

# text = """Dataset: rel-trial, Table: studies, source: Facet Biotech, is_fda_regulated_drug: None, plan_to_share_ipd: None, enrollment_type: None, is_us_export: None, brief_summaries: The purpose of The PROSPECT Study is to evaluate an investigational medication for the treatment of moderate to severe ulcerative colitis. This study is being conducted at up to 38 clinical research centers in the US, Canada, and Belgium, and is open to male and female patients 12 years and older. Participants in the study will have a number of visits to a research center over a five-month period. All study related care and medication is provided to qualified participants at no cost: this includes all visits, examinations, and laboratory work., fdaaa801_violation: None, brief_title: Humanized Anti-IL-2 Receptor Monoclonal Antibody in Moderate-to-severe Ulcerative Colitis, biospec_description: None, detailed_descriptions: None, acronym: None, phase: Phase 2, is_unapproved_device: None, official_title: A Phase II, Randomized, Double-blind, Placebo-controlled, Multi-center, Dose-ranging Study of Intravenous Daclizumab in Patients With Moderate-to-severe Ulcerative Colitis, nct_id: 7131, is_fda_regulated_device: None, baseline_type_units_analyzed: None, is_ppsd: None, enrollment: <-0.01345>, start_date: 2003-04-30 00:00:00, target_d number_of_arms: <nan>, baseline_population: None"""

# tokens = tokenizer.encode(text)
# print("Number of tokens:", len(tokens))



import relbench
from relbench.datasets import get_dataset
import os
import json


dataset = get_dataset(name="rel-amazon", download=True)

db = dataset.get_db()
print (db.table_dict.keys())

product_table = db.table_dict["product"]


# To see the actual data as a pandas DataFrame
product_df = product_table.df
print(product_df.head())

# Get the top 10 rows and save to CSV
top_10_products = product_df.head(10)
top_10_products.to_csv("top_10_products.csv", index=False)