import numpy as np
import pandas as pd
from scipy.stats import kendalltau

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if not r.size:
        return 0.0
    return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(y_true, y_score, k=10):
    """Compute nDCG@K for graded relevance in {0, 1, 2, 3}."""
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order)
    
    actual_dcg = dcg_at_k(y_true_sorted, k)
    ideal_dcg = dcg_at_k(sorted(y_true, reverse=True), k)
    
    if ideal_dcg == 0:
        return 1.0 if actual_dcg == 0 else 0.0
    return actual_dcg / ideal_dcg

def map_at_k(y_true, y_score, k=10, relevance_threshold=2):
    """Compute AP@K with binary relevance defined as grade >= 2."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if k <= 0 or y_true.size == 0:
        return 0.0
    relevant = y_true >= relevance_threshold
    total_relevant = int(relevant.sum())
    if total_relevant == 0:
        return 0.0

    order = np.argsort(-y_score, kind="mergesort")[:k]
    ranked_relevant = relevant[order]
    ranks = np.arange(1, ranked_relevant.size + 1)
    precision = np.cumsum(ranked_relevant) / ranks
    denominator = min(total_relevant, k)
    return float(np.sum(precision * ranked_relevant) / denominator)

def mrr_score(y_true, y_score):
    """
    Computes Mean Reciprocal Rank (MRR).
    """
    order = np.argsort(y_score)[::-1]
    y_true_sorted = (np.take(y_true, order) >= 1).astype(int)
    
    rel_idx = np.where(y_true_sorted == 1)[0]
    if len(rel_idx) == 0:
        return 0.0
    return 1.0 / (rel_idx[0] + 1.0)

def evaluate_models_on_dataset(models_dict, df_eval, target_col='gold_relevance', k_list=[5, 10]):
    """
    Evaluates a dictionary of models across all jobs in df_eval.
    Returns summary metrics table.
    """
    results = []
    job_ids = df_eval['job_id'].unique()
    
    for model_name, model in models_dict.items():
        # Get predictions
        if hasattr(model, 'predict'):
            scores = model.predict(df_eval)
        else:
            scores = model(df_eval)
            
        df_eval_copy = df_eval.copy()
        df_eval_copy['score'] = scores
        
        ndcg_5_list, ndcg_10_list, map_10_list, mrr_list = [], [], [], []
        
        for job_id, group in df_eval_copy.groupby('job_id'):
            y_t = group[target_col].values
            y_s = group['score'].values
            
            ndcg_5_list.append(ndcg_at_k(y_t, y_s, k=5))
            ndcg_10_list.append(ndcg_at_k(y_t, y_s, k=10))
            map_10_list.append(map_at_k(y_t, y_s, k=10))
            mrr_list.append(mrr_score(y_t, y_s))
            
        results.append({
            'Model': model_name,
            'nDCG@5': np.mean(ndcg_5_list),
            'nDCG@10': np.mean(ndcg_10_list),
            'MAP@10': np.mean(map_10_list),
            'MRR': np.mean(mrr_list)
        })
        
    return pd.DataFrame(results)

def per_job_ranking_metrics(
    df_eval,
    scores,
    target_col="relevance",
    k_list=(5, 10),
):
    """Return one row per job so aggregation and bootstrap preserve JD pairing."""
    frame = df_eval[["job_id", target_col]].copy()
    frame["score"] = np.asarray(scores, dtype=float)
    rows = []
    for job_id, group in frame.groupby("job_id", sort=True):
        row = {"job_id": job_id}
        for k in k_list:
            row[f"ndcg@{k}"] = ndcg_at_k(
                group[target_col].to_numpy(), group["score"].to_numpy(), k
            )
            row[f"map@{k}"] = map_at_k(
                group[target_col].to_numpy(), group["score"].to_numpy(), k
            )
        rows.append(row)
    return pd.DataFrame(rows)


def per_job_ndcg(df_eval, scores, target_col="gold_relevance", k=5):
    result = per_job_ranking_metrics(df_eval, scores, target_col, (k,))
    return result[["job_id", f"ndcg@{k}"]]


def paired_job_bootstrap(
    per_job_metrics,
    model_a,
    model_b,
    metric_col,
    model_col="model",
    job_col="job_id",
    n_bootstraps=1000,
    seed=42,
):
    """Bootstrap paired model differences by resampling complete job IDs."""
    subset = per_job_metrics[
        per_job_metrics[model_col].isin([model_a, model_b])
    ][[job_col, model_col, metric_col]]
    pivot = subset.pivot(index=job_col, columns=model_col, values=metric_col)
    if model_a not in pivot or model_b not in pivot:
        raise ValueError("Both models must have metrics for paired bootstrap")
    pivot = pivot[[model_a, model_b]].dropna()
    if pivot.empty:
        raise ValueError("No paired jobs available for bootstrap")

    differences = (pivot[model_b] - pivot[model_a]).to_numpy(float)
    rng = np.random.RandomState(seed)
    bootstrap_means = np.empty(n_bootstraps, dtype=float)
    for index in range(n_bootstraps):
        sampled_indices = rng.randint(0, len(differences), size=len(differences))
        bootstrap_means[index] = differences[sampled_indices].mean()
    return {
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric_col,
        "n_jobs": len(differences),
        "mean_delta": float(differences.mean()),
        "ci_95_low": float(np.percentile(bootstrap_means, 2.5)),
        "ci_95_high": float(np.percentile(bootstrap_means, 97.5)),
    }

def feature_error_slices(df_eval, scores, target_col='gold_relevance', k=5):
    """Diagnostic only: summarizes top-k errors by observable mismatch type."""
    frame = df_eval.copy(); frame["score"] = scores
    rows = []
    for job, g in frame.groupby("job_id"):
        top = g.nlargest(k, "score")
        for name, mask in {
            "skill_mismatch": top.get("missing_required_skill_ratio", pd.Series(0, index=top.index)) > .5,
            "experience_mismatch": top.get("experience_gap", pd.Series(0, index=top.index)) > 1,
            "role_mismatch": top.get("role_match", pd.Series(1, index=top.index)) < .1,
        }.items():
            rows.append({"job_id": job, "slice": name, "top_k_items": int(mask.sum()),
                         "mean_relevance": float(top.loc[mask, target_col].mean()) if mask.any() else np.nan})
    return pd.DataFrame(rows)

def paired_bootstrap_test(df_eval, model_a, model_b, metric_fn=ndcg_at_k, k=10, target_col='gold_relevance', n_bootstraps=1000, seed=42):
    """
    Performs Paired Bootstrap Resampling (B=1,000) over jobs in df_eval.
    Computes delta metric = Metric(Model B) - Metric(Model A).
    Returns mean delta, 95% Confidence Interval [CI_low, CI_high], and p-value.
    """
    rng = np.random.RandomState(seed)
    
    df_eval_copy = df_eval.copy()
    df_eval_copy['score_a'] = model_a.predict(df_eval)
    df_eval_copy['score_b'] = model_b.predict(df_eval)
    
    # Pre-calculate per-job metric scores for model A and model B
    job_metrics_a = {}
    job_metrics_b = {}
    
    for j_id, group in df_eval_copy.groupby('job_id'):
        y_t = group[target_col].values
        job_metrics_a[j_id] = metric_fn(y_t, group['score_a'].values, k=k)
        job_metrics_b[j_id] = metric_fn(y_t, group['score_b'].values, k=k)
        
    jobs = np.array(list(job_metrics_a.keys()))
    n_jobs = len(jobs)
    
    deltas = []
    for _ in range(n_bootstraps):
        boot_jobs = rng.choice(jobs, size=n_jobs, replace=True)
        m_a = np.mean([job_metrics_a[j] for j in boot_jobs])
        m_b = np.mean([job_metrics_b[j] for j in boot_jobs])
        deltas.append(m_b - m_a)
        
    deltas = np.array(deltas)
    mean_delta = np.mean(deltas)
    ci_low = np.percentile(deltas, 2.5)
    ci_high = np.percentile(deltas, 97.5)
    
    p_value = np.mean(deltas <= 0)
    
    return {
        'mean_delta': mean_delta,
        'ci_95_low': ci_low,
        'ci_95_high': ci_high,
        'p_value': p_value,
        'is_significant': ci_low > 0.0 and p_value < 0.05
    }

def circularity_divergence(df_res_heuristic, df_res_gold, metric_name='nDCG@10'):
    """
    Computes Circularity Divergence Index D_circ = 1 - tau_b(R_heuristic, R_gold).
    Quantifies how much circular evaluation distorts model rankings compared to gold evaluation.
    """
    ranks_heur = df_res_heuristic[metric_name].rank(ascending=False).values
    ranks_gold = df_res_gold[metric_name].rank(ascending=False).values
    
    tau, p_val = kendalltau(ranks_heur, ranks_gold)
    d_circ = 1.0 - tau
    
    return {
        'tau_b': tau,
        'kendall_p_val': p_val,
        'd_circ': d_circ,
        'has_rank_reversal': d_circ > 0.3
    }

def apply_holm_bonferroni_correction(test_results_dict, alpha=0.05):
    """
    Applies Holm-Bonferroni step-down correction to control Family-Wise Error Rate (FWER)
    across multiple hypothesis tests (H1, H2, Proposed D+).
    """
    tests = list(test_results_dict.keys())
    p_vals = [test_results_dict[t]['p_value'] for t in tests]
    
    sorted_indices = np.argsort(p_vals)
    m = len(tests)
    
    adjusted_results = {}
    for rank, idx in enumerate(sorted_indices):
        test_name = tests[idx]
        raw_p = p_vals[idx]
        adjusted_threshold = alpha / (m - rank)
        is_sig = raw_p < adjusted_threshold
        
        res = test_results_dict[test_name].copy()
        res['p_adjusted_threshold'] = adjusted_threshold
        res['is_significant_holm'] = is_sig
        adjusted_results[test_name] = res
        
    return adjusted_results
