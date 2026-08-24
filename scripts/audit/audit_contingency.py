import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from src.data import load_dataset, CVJobDatasetLoader
from src.weak import AspectLabelingFunctions
import itertools

def audit_data():
    print("Loading data with FIXED constructs...")
    df = load_dataset(data_dir="data", random_seed=42)
    loader = CVJobDatasetLoader(random_seed=42)
    df_train, _, _ = loader.get_job_disjoint_splits(df)
    
    # Check zero mass on skill_iou
    zero_skill_ratio = (df_train['skill_iou'] == 0.0).mean()
    print(f"\n[Sparsity Audit] Fraction of train pairs with EXACTLY zero skill_iou: {zero_skill_ratio:.2%}")
    
    lf = AspectLabelingFunctions(pos_percentile=75, neg_percentile=25)
    lf.fit(df_train)
    df_train = lf.transform(df_train)
    
    # Correlation Matrix
    print("\n--- LF Correlation Matrix (Pearson) ---")
    lf_cols = ['lf_skill', 'lf_sem', 'lf_role', 'lf_loc']
    corr = df_train[lf_cols].corr().round(3)
    print(corr.to_string())
    
    # Contingency table of LF patterns
    print("\n--- Contingency Table (Top 15 LF Patterns) ---")
    
    def get_pattern(row):
        mapping = {1: '+', 0: '0', -1: '-'}
        return "".join([mapping.get(row[c], '0') for c in lf_cols])
        
    df_train['lf_pattern'] = df_train.apply(get_pattern, axis=1)
    pattern_counts = df_train['lf_pattern'].value_counts().head(15)
    
    print("Pattern (Skill, Sem, Role, Loc) | Count | % of Train")
    print("-" * 52)
    for pat, count in pattern_counts.items():
        pct = count / len(df_train)
        print(f"{pat:^31} | {count:^5} | {pct:.1%}")
        
    # Check Bucket A specific contamination
    # Bucket A: lf_skill==1 & lf_sem==1 & lf_role==1 & lf_loc==1
    bucket_a = df_train[(df_train['lf_skill']==1) & (df_train['lf_sem']==1) & (df_train['lf_role']==1) & (df_train['lf_loc']==1)]
    print(f"\n--- Bucket A Analysis ---")
    print(f"Bucket A total pairs: {len(bucket_a)}")
    
    # Check completely disjoint job title and desired job
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    
    def tokenize(text):
        if pd.isna(text): return set()
        import re
        tokens = re.findall(r'\b\w+\b', str(text).lower())
        return set(tokens) - ENGLISH_STOP_WORDS - {'nhân', 'viên', 'nhan', 'vien', 'chuyên', 'chuyen', 'thực', 'tập', 'sinh', 'công', 'cong', 'việc', 'viec', 'kỹ', 'ky', 'năng', 'nang'}

    disjoint_count = 0
    df_j = pd.read_csv('data/JOB_DATA_FINAL.csv')
    df_u = pd.read_csv('data/USER_DATA_FINAL.csv')
    
    # We need to map pair_id to original text. Easiest way is to do the same filtering as load_text_data.
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
            if disjoint_count <= 3:
                print(f"[Disjoint Warning] pair_id={row['pair_id']}")
                print(f"  JD Title: {j_title} -> {j_toks}")
                print(f"  CV Desired: {u_title} -> {u_toks}")
                
    print(f"Bucket A pairs with ZERO non-boilerplate title token overlap: {disjoint_count} ({disjoint_count/max(1,len(bucket_a)):.1%})")

    print("\n--- Regression Check: Pair 5523 (CSKH vs Lái xe) ---")
    suspect = df_train[df_train['pair_id'] == 5523]
    if len(suspect) > 0:
        row = suspect.iloc[0]
        print(f"pair_id: 5523")
        print(f"job_title: {row['job_title']}")
        print(f"skill_iou: {row['skill_iou']:.4f} -> lf_skill: {row['lf_skill']}")
        print(f"role_match: {row['role_match']:.4f} -> lf_role: {row['lf_role']}")
        print(f"desc_sem_sim: {row['desc_sem_sim']:.4f} -> lf_sem: {row['lf_sem']}")
        print(f"loc_match: {row['loc_match']} -> lf_loc: {row['lf_loc']}")
        
if __name__ == "__main__":
    audit_data()
