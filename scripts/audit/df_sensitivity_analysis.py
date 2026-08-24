import pandas as pd
import numpy as np
import json
import sys

from src.data import load_dataset, CVJobDatasetLoader
from src.weak import AspectLabelingFunctions

def run_sensitivity():
    print("Loading Global DF Dictionary...")
    try:
        with open('data/global_skill_df.json', 'r', encoding='utf-8') as f:
            global_df = json.load(f)
    except FileNotFoundError:
        print("Please run compute_global_df.py first.")
        sys.exit(1)
        
    print("\nStarting Sensitivity Analysis...")
    thresholds = [1.0, 0.20, 0.15, 0.10, 0.05]
    
    results = []
    regression_pairs = [5523, 703, 743] # Add some other boilerplate heavy pairs
    
    for tau in thresholds:
        print(f"\n=======================================================")
        print(f"--- Testing tau = {tau if tau < 1.0 else '∞'} ---")
        
        # Load dataset with current tau
        df_train = load_dataset(data_dir='data', random_seed=42, df_threshold=tau)
        loader = CVJobDatasetLoader(random_seed=42)
        df_train, _, _ = loader.get_job_disjoint_splits(df_train)
        
        lf = AspectLabelingFunctions(pos_percentile=75, neg_percentile=25)
        lf.fit(df_train)
        df_train = lf.transform(df_train)
        
        # Metrics
        skill_iou = df_train['skill_iou']
        median_iou = skill_iou.median()
        p75 = np.percentile(skill_iou, 75)
        p90 = np.percentile(skill_iou, 90)
        coverage = (df_train['lf_skill'] != 0).mean()
        
        bucket_a = df_train[(df_train['lf_skill'] == 1) & 
                            (df_train['lf_sem'] == 1) & 
                            (df_train['lf_role'] == 1) & 
                            (df_train['lf_loc'] == 1)]
        
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        def tokenize(text):
            if pd.isna(text): return set()
            import re
            tokens = re.findall(r'\b\w+\b', str(text).lower())
            return set(tokens) - ENGLISH_STOP_WORDS - {'nhân', 'viên', 'nhan', 'vien', 'chuyên', 'chuyen', 'thực', 'tập', 'sinh', 'công', 'cong', 'việc', 'viec', 'kỹ', 'ky', 'năng', 'nang'}

        disjoint_count = 0
        df_j = pd.read_csv('data/JOB_DATA_FINAL.csv')
        df_u = pd.read_csv('data/USER_DATA_FINAL.csv')
        jobs = df_j.dropna(subset=['Job Title', 'Job Requirements']).sample(n=min(len(df_j), 80), random_state=42).reset_index(drop=True)
        users = df_u.dropna(subset=['Desired Job', 'Skills']).sample(n=min(len(df_u), 120), random_state=42).reset_index(drop=True)
        
        for _, row in bucket_a.iterrows():
            j_idx = int(row['job_id'].split('_')[1])
            u_idx = int(row['cand_id'].split('_')[1])
            j_title = str(jobs.loc[j_idx, 'Job Title'])
            u_title = str(users.loc[u_idx, 'Desired Job'])
            j_toks = tokenize(j_title)
            u_toks = tokenize(u_title)
            if len(j_toks.intersection(u_toks)) == 0:
                disjoint_count += 1
                
        false_consensus = disjoint_count / len(bucket_a) if len(bucket_a) > 0 else 0
        
        results.append({
            'tau': tau if tau < 1.0 else '∞',
            'median_iou': median_iou,
            'p75': p75,
            'p90': p90,
            'coverage': coverage,
            'bucket_a': len(bucket_a),
            'false_consensus': false_consensus
        })
        
        print("\n[Regression Cases]")
        for pid in regression_pairs:
            suspect = df_train[df_train['pair_id'] == pid]
            if not suspect.empty:
                row = suspect.iloc[0]
                print(f"pair_id {pid} -> skill_iou: {row['skill_iou']:.4f} (lf_skill: {row['lf_skill']})")
                
    print("\n\n--- Final Sensitivity Analysis Results ---")
    print(f"{'tau':>5} | {'Median IoU':>10} | {'P75':>10} | {'P90':>10} | {'Coverage':>10} | {'Bucket A':>10} | {'False Consensus':>15}")
    print("-" * 88)
    for r in results:
        tau_str = str(r['tau'])
        print(f"{tau_str:>5} | {r['median_iou']:>10.4f} | {r['p75']:>10.4f} | {r['p90']:>10.4f} | {r['coverage']:>9.1%} | {r['bucket_a']:>10} | {r['false_consensus']:>14.1%}")

if __name__ == "__main__":
    run_sensitivity()
