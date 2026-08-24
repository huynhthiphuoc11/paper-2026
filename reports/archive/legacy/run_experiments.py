import sys
import numpy as np
import pandas as pd

from src.data import CVJobDatasetLoader, load_dataset
from src.weak_supervision import WeakSupervisionFramework
from src.models import (
    ModelH_Heuristic,
    ModelA_FixedBCE,
    ModelB_LearnedBCE,
    ModelB_Plus_SoftBCE,
    ModelC_FixedRankNet,
    ModelD_LearnedRankNet,
    ModelD_Plus_ProposedSoftRankNet
)
from src.evaluation import (
    evaluate_models_on_dataset,
    paired_bootstrap_test,
    circularity_divergence,
    apply_holm_bonferroni_correction
)

def main():
    print("=" * 80)
    print(" EXPERIMENT PIPELINE: VIETNAMESE CV-JOB RANKING (UNBIASED & LTR)")
    print("=" * 80)
    
    # 1. Load Data (Supports Kaggle JOB_DATA_FINAL.csv in data/ or root)
    print("\n[1/5] Loading & Partitioning Dataset (Checking data/JOB_DATA_FINAL.csv)...")
    loader_obj = CVJobDatasetLoader(random_seed=42)
    df_raw = load_dataset(data_dir='data', random_seed=42)
    df_train, df_dev, df_test = loader_obj.get_job_disjoint_splits(df_raw)
    
    print(f" -> Train Set (Job-Disjoint): {len(df_train)} pairs across {df_train['job_id'].nunique()} Jobs.")
    print(f" -> Dev Set (Gold Annotated) : {len(df_dev)} pairs across {df_dev['job_id'].nunique()} Jobs.")
    print(f" -> Test Set (Gold Annotated): {len(df_test)} pairs across {df_test['job_id'].nunique()} Jobs.")
    
    # 2. Weak Supervision Setup
    print("\n[2/5] Running Heterogeneous Weak Supervision & Quality Checks...")
    ws = WeakSupervisionFramework(sem_pct=70, skill_pct=70, exp_pct=60)
    
    df_train_weak = ws.generate_weak_signals(df_train)
    df_dev_weak = ws.generate_weak_signals(df_dev)
    
    kappa = ws.compute_fleiss_kappa(df_train_weak)
    dev_precisions = ws.evaluate_precision_on_dev(df_dev_weak)
    
    print(f" -> Inter-Source Fleiss' Kappa: {kappa:.4f}")
    for lf_name, prec in dev_precisions.items():
        print(f"    * {lf_name} Precision on Dev-Gold: {prec*100:.2f}%")
        
    print("\n -> Fitting Dawid-Skene EM Generative Label Model...")
    df_train_weak = ws.predict_probabilistic_labels(df_train_weak)
    print(f"    Estimated Source Sensitivities (alpha): {ws.source_sensitivities}")
    print(f"    Estimated Source Specificities (beta) : {ws.source_specificities}")
    
    # 3. Model Training (Ablation H -> D+)
    print("\n[3/5] Training Models in Controlled Ablation Matrix (H, A, B, B+, C, D, D+)...")
    
    models = {
        'H (Heuristic)': ModelH_Heuristic(),
        'A (Baseline BCE)': ModelA_FixedBCE(),
        'B (Learned BCE)': ModelB_LearnedBCE(),
        'B+ (Soft BCE)': ModelB_Plus_SoftBCE(),
        'C (Fixed RankNet)': ModelC_FixedRankNet(epochs=30),
        'D (Main RankNet)': ModelD_LearnedRankNet(epochs=40),
        'D+ (Proposed Soft-RankNet)': ModelD_Plus_ProposedSoftRankNet(epochs=50)
    }
    
    # Train trainable models
    models['A (Baseline BCE)'].fit(df_train_weak)
    models['B (Learned BCE)'].fit(df_train_weak)
    models['B+ (Soft BCE)'].fit(df_train_weak)
    models['C (Fixed RankNet)'].fit(df_train_weak)
    models['D (Main RankNet)'].fit(df_train_weak)
    models['D+ (Proposed Soft-RankNet)'].fit(df_train_weak)
    print(" -> All models trained successfully.")
    
    # Sanity Check: Verify Monotonic Invariance of Model A vs Model C
    res_gold = evaluate_models_on_dataset(models, df_test, target_col='gold_relevance')
    res_heur = evaluate_models_on_dataset(models, df_test, target_col='heuristic_score')
    
    print("\n" + "=" * 65)
    print("  TABLE 1: MODEL PERFORMANCE ON INDEPENDENT GOLD SET (JOB-DISJOINT)")
    print("=" * 65)
    print(res_gold.to_string(index=False))
    
    print("\n" + "=" * 65)
    print("  TABLE 2: MODEL PERFORMANCE ON HEURISTIC TEST SET (CIRCULAR EVAL)")
    print("=" * 65)
    print(res_heur.to_string(index=False))
    
    # 5. Statistical Significance Testing & Circularity Divergence
    print("\n[5/5] Performing Paired Bootstrap Resampling (B=1,000) & Holm-Bonferroni Correction...")
    
    raw_tests = {
        'H1 (Weight Learning : B vs A)': paired_bootstrap_test(df_test, models['A (Baseline BCE)'], models['B (Learned BCE)'], k=10),
        'H2 (Pairwise LTR    : D vs B)': paired_bootstrap_test(df_test, models['B (Learned BCE)'], models['D (Main RankNet)'], k=10),
        'Proposed Soft-RankNet: D+ vs A': paired_bootstrap_test(df_test, models['A (Baseline BCE)'], models['D+ (Proposed Soft-RankNet)'], k=10)
    }
    
    corrected_tests = apply_holm_bonferroni_correction(raw_tests, alpha=0.05)
    
    print("\n" + "=" * 80)
    print("  TABLE 3: STATISTICAL HYPOTHESIS TESTING (HOLM-BONFERRONI ADJUSTED)")
    print("=" * 80)
    for test_name, res in corrected_tests.items():
        sig_str = "(Sig. Holm-Bonferroni)" if res['is_significant_holm'] else "(Not Significant)"
        print(f"{test_name:30s} -> Delta nDCG@10 = {res['mean_delta']:+.4f} [95% CI: {res['ci_95_low']:.4f}, {res['ci_95_high']:.4f}], p = {res['p_value']:.4f} {sig_str}")
        
    # Hypothesis H3: Circularity Divergence Metric
    circ_info = circularity_divergence(res_heur, res_gold, metric_name='nDCG@10')
    print("\n" + "=" * 80)
    print("  TABLE 4: CIRCULARITY DIVERGENCE INDEX (H3 VERIFICATION)")
    print("=" * 80)
    print(f"Kendall's Tau_b (Heuristic Ranks vs Gold Ranks): {circ_info['tau_b']:.4f} (p = {circ_info['kendall_p_val']:.4f})")
    print(f"Circularity Divergence Index (D_circ)          : {circ_info['d_circ']:.4f}")
    print(f"Rank Reversal Confirmed (D_circ > 0.30)        : {circ_info['has_rank_reversal']}")
    print("=" * 80)
    print("\nExperiment Pipeline Completed Successfully!")

if __name__ == '__main__':
    main()
