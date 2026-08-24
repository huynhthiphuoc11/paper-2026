"""Quick feature distribution check — run to diagnose root cause of lf_exp degeneration."""
import sys, os
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from src.data import load_dataset, CVJobDatasetLoader

df = load_dataset(data_dir="data", random_seed=42)
loader = CVJobDatasetLoader(random_seed=42)
df_train, _, _ = loader.get_job_disjoint_splits(df)

print("=== exp_score distribution ===")
print(df_train["exp_score"].describe())
print()
n_exact_1 = (df_train["exp_score"] == 1.0).sum()
print(f"exp_score == 1.0 exactly: {n_exact_1} ({n_exact_1/len(df_train):.1%})")
print()
for p in [10, 25, 50, 75, 90, 95, 99]:
    val = float(np.percentile(df_train["exp_score"], p))
    print(f"  p{p:2d}: {val:.4f}")

print()
print("=== skill_iou stats ===")
print(df_train["skill_iou"].describe())

print()
print("=== loc_match value counts ===")
print(df_train["loc_match"].value_counts(normalize=True))

print()
print("=== desc_sem_sim stats ===")
print(df_train["desc_sem_sim"].describe())
