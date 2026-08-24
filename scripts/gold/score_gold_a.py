"""
Score Gold-A Annotation Results
================================
- Checks consistency of labels vs gating answers
- Calculates Cohen's Kappa per bucket
- Computes Wilson 95% CIs for Bucket C and C'
- Outputs the architectural decision (3-LF vs 4-LF, RankNet feature design)
"""

import sys
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.metrics import cohen_kappa_score

def wilson_ci(k, n, alpha=0.05):
    if n == 0: return (0.0, 1.0)
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    half   = z * ((p*(1-p)/n + z**2/(4*n**2))**0.5) / denom
    return round(centre - half, 3), round(centre + half, 3)

def check_consistency(row):
    label_str = str(row['label']).strip()
    q2_str = str(row['Q2']).strip().lower()
    
    # Map label to 1/0
    if label_str == '1': label = 1
    elif label_str == '0': label = 0
    else: return 'MISSING_LABEL'
    
    # Map dealbreaker to Yes/No/Uncertain
    if 'yes' in q2_str: gating = 'Yes'
    elif 'no' in q2_str: gating = 'No'
    else: gating = 'Uncertain'
    
    # Inconsistencies
    if label == 1 and gating == 'Yes': return 'INCONSISTENT_RELEVANT_BUT_DEALBREAKER'
    if label == 0 and gating == 'No': return 'INCONSISTENT_IRRELEVANT_BUT_NOT_DEALBREAKER'
    
    return 'CONSISTENT'

def score_gold_a(annotator_1_file, annotator_2_file, master_key_file):
    print("Loading annotation files...")
    try:
        a1 = pd.read_csv(annotator_1_file)
        a2 = pd.read_csv(annotator_2_file)
        mk = pd.read_csv(master_key_file)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Ensure generate_annotation_batches.py has been run and annotators have filled the files.")
        return
        
    a1 = a1.rename(columns={'label (1=relevant, 0=not_relevant)': 'label', 'Q2_dealbreaker (Yes/No/Uncertain)': 'Q2'})
    a2 = a2.rename(columns={'label (1=relevant, 0=not_relevant)': 'label', 'Q2_dealbreaker (Yes/No/Uncertain)': 'Q2'})
    
    df = pd.merge(mk[['pair_id', 'bucket']], a1[['pair_id', 'label', 'Q2']], on='pair_id', suffixes=('', '_a1'))
    df = pd.merge(df, a2[['pair_id', 'label', 'Q2']], on='pair_id', suffixes=('_a1', '_a2'))
    
    # Process each annotator's consistency
    df['consistency_a1'] = df.apply(lambda r: check_consistency({'label': r['label_a1'], 'Q2': r['Q2_a1']}), axis=1)
    df['consistency_a2'] = df.apply(lambda r: check_consistency({'label': r['label_a2'], 'Q2': r['Q2_a2']}), axis=1)
    
    # Final label resolution (simple logic for script: must agree and be consistent)
    # In reality, a 3rd annotator resolves conflicts. Here we simulate the pipeline requirement.
    def resolve_label(row):
        if row['consistency_a1'] != 'CONSISTENT' or row['consistency_a2'] != 'CONSISTENT':
            return np.nan # Needs resolution
        if row['label_a1'] != row['label_a2']:
            return np.nan # Conflict
        return int(row['label_a1'])
        
    df['final_label'] = df.apply(resolve_label, axis=1)
    
    print("\n--- Cohen's Kappa per Bucket ---")
    buckets = ['A', 'B', 'C', 'C_prime', 'D', 'E']
    for b in buckets:
        sub = df[df['bucket'] == b]
        if len(sub) == 0: continue
        
        # Valid pairs where both provided 0 or 1
        sub_valid = sub[sub['label_a1'].isin(['0', '1', 0, 1]) & sub['label_a2'].isin(['0', '1', 0, 1])]
        if len(sub_valid) < len(sub):
            print(f"Bucket {b}: {len(sub) - len(sub_valid)} missing/invalid labels")
            
        if len(sub_valid) > 0:
            l1 = sub_valid['label_a1'].astype(int)
            l2 = sub_valid['label_a2'].astype(int)
            kappa = cohen_kappa_score(l1, l2)
            print(f"Bucket {b:<8} (n={len(sub_valid):<3}): κ = {kappa:.3f}")
        else:
            print(f"Bucket {b:<8} (n=0  ): κ = N/A")
            
    # Decision Logic for C and C'
    print("\n--- Decision Analysis ---")
    
    # Bucket C
    c_df = df[df['bucket'] == 'C']
    c_valid = c_df.dropna(subset=['final_label'])
    n_c = len(c_valid)
    
    print(f"\nBucket C (loc mismatch vs strong skills):")
    print(f"  Total pairs: {len(c_df)}")
    print(f"  Consistent & Agreed labels: {n_c}")
    
    c_gates = False
    c_ambiguous = True
    if n_c > 0:
        k_c = int(c_valid['final_label'].sum())
        p_c = k_c / n_c
        lo_c, hi_c = wilson_ci(k_c, n_c)
        print(f"  Relevant (soft loc): {k_c}/{n_c} ({p_c:.1%})")
        print(f"  Wilson 95% CI: [{lo_c:.3f}, {hi_c:.3f}] (Threshold 0.70)")
        
        if lo_c > 0.70:
            c_gates = False
            c_ambiguous = False
            print("  Conclusion: CLEAR KEEP (Location is soft evidence, DOES NOT gate)")
        elif hi_c < 0.70:
            c_gates = True
            c_ambiguous = False
            print("  Conclusion: CLEAR REMOVE (Location mismatch DOES gate)")
        else:
            print("  Conclusion: AMBIGUOUS (CI straddles 0.70)")

    # Bucket C'
    cp_df = df[df['bucket'] == 'C_prime']
    cp_valid = cp_df.dropna(subset=['final_label'])
    n_cp = len(cp_valid)
    
    print(f"\nBucket C' (loc match vs weak skills):")
    print(f"  Total pairs: {len(cp_df)}")
    print(f"  Consistent & Agreed labels: {n_cp}")
    
    cp_rescues = False
    cp_ambiguous = True
    if n_cp > 0:
        k_cp = int(cp_valid['final_label'].sum())
        p_cp = k_cp / n_cp
        lo_cp, hi_cp = wilson_ci(k_cp, n_cp)
        print(f"  Relevant (loc rescues): {k_cp}/{n_cp} ({p_cp:.1%})")
        print(f"  Wilson 95% CI: [{lo_cp:.3f}, {hi_cp:.3f}] (Threshold 0.50)")
        
        if lo_cp > 0.50:
            cp_rescues = True
            cp_ambiguous = False
            print("  Conclusion: CLEAR RESCUE (Location match DOES rescue poor skills)")
        elif hi_cp < 0.50:
            cp_rescues = False
            cp_ambiguous = False
            print("  Conclusion: CLEAR NO-RESCUE (Location match does NOT rescue poor skills)")
        else:
            print("  Conclusion: AMBIGUOUS (CI straddles 0.50)")

    print("\n--- Final Architectural Decision ---")
    if c_ambiguous:
        print("RESULT: Bucket C is AMBIGUOUS.")
        print("ACTION: Run Stage 4 with BOTH 3-LF and 4-LF Label Models in parallel.")
        print("        Final choice decided by Gold-B-dev F1.")
    elif c_gates and not cp_rescues:
        print("RESULT: Location is a HARD NECESSARY CONDITION, not sufficient.")
        print("ACTION: REMOVE lf_loc from Label Model (use K=3: skill, sem, role).")
        print("        Add lf_loc as pre-ranking filter / RankNet feature (penalty-only).")
    elif not c_gates and not cp_rescues:
        print("RESULT: Location is TRULY SOFT EVIDENCE.")
        print("ACTION: KEEP 4 LFs in Label Model.")
    elif c_gates and cp_rescues:
        print("RESULT: Location is a SYMMETRIC GATE (Necessary AND Sufficient).")
        print("ACTION: REMOVE lf_loc from Label Model.")
        print("        RankNet feature uses penalty + bonus.")
    else:
        print("RESULT: Location is Sufficient but not Necessary (Rare).")
        print("ACTION: KEEP 4 LFs in Label Model or design custom RankNet feature.")
        
    if not c_ambiguous and cp_ambiguous:
        print("\nNote: Bucket C' was ambiguous. RankNet feature defaults to penalty-only (conservative) pending Gold-B.")

if __name__ == "__main__":
    if len(sys.argv) == 4:
        score_gold_a(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        # Default paths if run directly
        a1 = "data/gold/gold_a_batches/gold_a_annotator_1.csv"
        a2 = "data/gold/gold_a_batches/gold_a_annotator_2.csv"
        mk = "data/gold/gold_a_batches/gold_a_master_key.csv"
        score_gold_a(a1, a2, mk)
