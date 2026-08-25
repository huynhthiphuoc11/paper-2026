from __future__ import annotations

import numpy as np
import pandas as pd


def dcg_at_k(relevance, k: int) -> float:
    values = np.asarray(relevance, dtype=float)[:k]
    if not len(values):
        return 0.0
    return float(np.sum((2.0 ** values - 1.0) / np.log2(np.arange(2, len(values) + 2))))


def ndcg_at_k(relevance, scores, k: int) -> float:
    relevance = np.asarray(relevance, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    actual = dcg_at_k(relevance[order], k)
    ideal = dcg_at_k(np.sort(relevance)[::-1], k)
    return 0.0 if ideal == 0.0 else float(actual / ideal)


def reciprocal_rank(relevance, scores, threshold: int = 2) -> float:
    relevance = np.asarray(relevance, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    positions = np.flatnonzero(relevance[order] >= threshold)
    return 0.0 if not len(positions) else float(1.0 / (positions[0] + 1))


def per_job_metrics(
    frame: pd.DataFrame,
    scores,
    target_column: str = "relevance",
    k_values: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    evaluation = frame[["job_id", target_column]].copy()
    evaluation["score"] = np.asarray(scores, dtype=float)
    rows = []
    for job_id, group in evaluation.groupby("job_id", sort=True):
        row = {"job_id": job_id}
        for k in k_values:
            row[f"ndcg@{k}"] = ndcg_at_k(group[target_column], group["score"], k)
        row["mrr"] = reciprocal_rank(group[target_column], group["score"], threshold=2)
        rows.append(row)
    return pd.DataFrame(rows)


def macro_metrics(per_job: pd.DataFrame) -> dict:
    columns = [column for column in per_job.columns if column != "job_id"]
    return {column: float(per_job[column].mean()) for column in columns}


def paired_job_bootstrap(
    per_job: pd.DataFrame,
    baseline: str,
    proposed: str,
    metric: str,
    n_resamples: int,
    seed: int,
) -> dict:
    subset = per_job[per_job["system"].isin([baseline, proposed])]
    pivot = subset.pivot(index="job_id", columns="system", values=metric)[[baseline, proposed]].dropna()
    if pivot.empty:
        raise ValueError("No paired jobs for bootstrap")
    differences = (pivot[proposed] - pivot[baseline]).to_numpy(float)
    rng = np.random.RandomState(seed)
    means = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample = rng.randint(0, len(differences), size=len(differences))
        means[index] = differences[sample].mean()
    return {
        "baseline": baseline,
        "proposed": proposed,
        "metric": metric,
        "n_jobs": int(len(differences)),
        "mean_delta": float(differences.mean()),
        "ci_95_low": float(np.percentile(means, 2.5)),
        "ci_95_high": float(np.percentile(means, 97.5)),
        "supports_improvement": bool(np.percentile(means, 2.5) > 0.0),
    }


def select_formulation(
    validation_summary: pd.DataFrame,
    tie_tolerance: float = 0.005,
) -> str:
    required = {"formulation", "ndcg@5", "validation_pairwise_loss"}
    missing = required - set(validation_summary.columns)
    if missing:
        raise ValueError(
            "Formulation selection requires: " + ", ".join(sorted(missing))
        )
    means = validation_summary.groupby("formulation", as_index=False).agg(
        **{"ndcg@5": ("ndcg@5", "mean")},
        validation_pairwise_loss=("validation_pairwise_loss", "mean"),
    )
    best_value = float(means["ndcg@5"].max())
    tied = means.loc[means["ndcg@5"] >= best_value - tie_tolerance].copy()
    best_loss = float(tied["validation_pairwise_loss"].min())
    tied = set(tied.loc[
        tied["validation_pairwise_loss"] <= best_loss + 1e-12,
        "formulation",
    ])
    simplicity = ["pointwise", "pairwise", "listwise"]
    return next(name for name in simplicity if name in tied)
