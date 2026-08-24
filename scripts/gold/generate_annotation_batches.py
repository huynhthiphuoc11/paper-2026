"""
Generate Gold-A Annotation Batches
===================================
- Samples 191 pairs across 6 buckets (A, B, C, C', D, E)
- Blinds all LF values
- Merges raw CV and JD text from original datasets
- Shuffles everything so annotators don't know the bucket
- Generates 100% overlapping assignments for 2 annotators
"""

import os
import random
import pandas as pd
import sys
sys.path.insert(0, '.')
from src.data import load_dataset, CVJobDatasetLoader
from src.weak import AspectLabelingFunctions

def load_text_data():
    df_j = pd.read_csv('data/JOB_DATA_FINAL.csv')
    df_u = pd.read_csv('data/USER_DATA_FINAL.csv')
    
    random_seed = 42
    n_jobs = 80
    n_candidates = 120
    
    jobs = df_j.dropna(subset=['Job Title', 'Job Requirements']).sample(n=min(len(df_j), n_jobs), random_state=random_seed).reset_index(drop=True)
    users = df_u.dropna(subset=['Desired Job', 'Skills']).sample(n=min(len(df_u), n_candidates), random_state=random_seed).reset_index(drop=True)
    
    # Clean text columns to avoid Excel/CSV formatting nightmares
    def clean_text(text):
        if pd.isna(text): return ""
        return str(text).replace('\r', ' ').replace('\n', ' | ').strip()
    
    jobs['jd_text'] = "TITLE: " + jobs['Job Title'].fillna("") + " | REQ: " + jobs['Job Requirements'].fillna("") + " | DESC: " + jobs['Job Description'].fillna("")
    jobs['jd_text'] = jobs['jd_text'].apply(clean_text)
    
    users['cv_text'] = "DESIRED JOB: " + users['Desired Job'].fillna("") + " | SKILLS: " + users['Skills'].fillna("") + " | EXP: " + users['Work Experience'].fillna("")
    users['cv_text'] = users['cv_text'].apply(clean_text)
    
    # Keys are 'JOB_XX' and 'CV_XXX' based on row index
    job_dict = {f"JOB_{i:02d}": text for i, text in enumerate(jobs['jd_text'])}
    user_dict = {f"CV_{i:03d}": text for i, text in enumerate(users['cv_text'])}
    
    return job_dict, user_dict

def generate_batches():
    print("Loading data and computing LFs...")
    df = load_dataset(data_dir="data", random_seed=42)
    loader = CVJobDatasetLoader(random_seed=42)
    df_train, _, _ = loader.get_job_disjoint_splits(df)
    
    lf = AspectLabelingFunctions(pos_percentile=75, neg_percentile=25)
    lf.fit(df_train)
    df_train = lf.transform(df_train)
    
    job_texts, cv_texts = load_text_data()
    
    # Define bucket filters
    buckets = {
        'A': (df_train['lf_skill']==1) & (df_train['lf_sem']==1) & (df_train['lf_role']==1) & (df_train['lf_loc']==1),
        'B': (df_train['lf_skill']==-1) & (df_train['lf_sem']==-1) & (df_train['lf_role']==-1) & (df_train['lf_loc']==-1),
        'C': (df_train['lf_skill']==1) & (df_train['lf_sem']==1) & (df_train['lf_role']==1) & (df_train['lf_loc']==-1),
        'C_prime': (df_train['lf_skill']==-1) & (df_train['lf_sem']==-1) & (df_train['lf_role']==-1) & (df_train['lf_loc']==1),
        'D': (df_train['lf_loc']==1) & 
             (((df_train['lf_skill']==1).astype(int) + (df_train['lf_sem']==1).astype(int) + (df_train['lf_role']==1).astype(int)) == 1) &
             (((df_train['lf_skill']==-1).astype(int) + (df_train['lf_sem']==-1).astype(int) + (df_train['lf_role']==-1).astype(int)) >= 1),
        'E': (df_train['lf_skill']==0) & (df_train['lf_sem']==0) & (df_train['lf_role']==0) & (df_train['lf_loc'] != 0)
    }
    
    targets = {'A': 20, 'B': 20, 'C': 61, 'C_prime': 40, 'D': 30, 'E': 20}
    
    sampled_pairs = []
    
    for b_name, condition in buckets.items():
        pool = df_train[condition]
        target = targets[b_name]
        
        # In bucket E, ensure balanced loc +1 and -1 if possible
        if b_name == 'E':
            pool_pos = pool[pool['lf_loc'] == 1]
            pool_neg = pool[pool['lf_loc'] == -1]
            samp_pos = pool_pos.sample(n=min(len(pool_pos), target//2), random_state=42)
            samp_neg = pool_neg.sample(n=min(len(pool_neg), target//2), random_state=42)
            samp = pd.concat([samp_pos, samp_neg])
            # fill rest if short
            if len(samp) < target:
                rem = pool.drop(samp.index)
                samp = pd.concat([samp, rem.sample(n=min(len(rem), target - len(samp)), random_state=42)])
        else:
            n_sample = min(len(pool), target)
            samp = pool.sample(n=n_sample, random_state=42)
            if n_sample < target:
                print(f"Warning: Bucket {b_name} pool has {n_sample} pairs (target {target})")
                
        for _, row in samp.iterrows():
            pair_id = row['pair_id']
            sampled_pairs.append({
                'pair_id': pair_id,
                'job_id': row['job_id'],
                'cand_id': row['cand_id'],
                'bucket': b_name,
                'jd_text': job_texts.get(row['job_id'], "NOT_FOUND"),
                'cv_text': cv_texts.get(row['cand_id'], "NOT_FOUND"),
                'lf_skill': row['lf_skill'],
                'lf_sem': row['lf_sem'],
                'lf_role': row['lf_role'],
                'lf_loc': row['lf_loc']
            })

    # Convert to df and shuffle
    df_all = pd.DataFrame(sampled_pairs)
    df_all = df_all.sample(frac=1, random_state=99).reset_index(drop=True)
    
    print(f"\nGenerated {len(df_all)} total pairs.")
    print("Bucket distribution:")
    print(df_all['bucket'].value_counts())
    
    os.makedirs('data/gold/gold_a_batches', exist_ok=True)
    
    # 1. Save Master Key (with LFs and Bucket ID)
    df_all.to_csv('data/gold/gold_a_batches/gold_a_master_key.csv', index=False)
    
    # 2. Save Blinded Annotation Files (100% overlap for Annotator 1 and 2)
    # Exclude bucket and LFs. Add blank columns for answers.
    blinded_cols = ['pair_id', 'jd_text', 'cv_text']
    df_blinded = df_all[blinded_cols].copy()
    
    # We ask Q2 for ALL pairs to not leak bucket information
    df_blinded['label (1=relevant, 0=not_relevant)'] = ""
    df_blinded['Q2_dealbreaker (Yes/No/Uncertain)'] = ""
    df_blinded['annotator_id'] = ""
    
    df_blinded.to_csv('data/gold/gold_a_batches/gold_a_annotator_1.csv', index=False)
    df_blinded.to_csv('data/gold/gold_a_batches/gold_a_annotator_2.csv', index=False)
    
    print("\n[Done] Batches saved to 'data/gold/gold_a_batches/'")
    print("  -> gold_a_master_key.csv (KEEP SECRET)")
    print("  -> gold_a_annotator_1.csv (Give to Annotator 1)")
    print("  -> gold_a_annotator_2.csv (Give to Annotator 2)")

if __name__ == "__main__":
    generate_batches()
