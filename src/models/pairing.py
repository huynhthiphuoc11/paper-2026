"""Auditable within-job pair construction for RankNet."""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd

from src.data.loader import FEATURE_COLS

LF_COLS = ["lf_skill", "lf_sem", "lf_exp", "lf_role", "lf_loc"]
PAIR_ID_COLS = [
    "job_id",
    "preferred_row_id",
    "nonpreferred_row_id",
    "preferred_cand_id",
    "nonpreferred_cand_id",
]


def _row_identifier(row: pd.Series, fallback: int):
    return row.get("pair_id", fallback)


def _has_lf_conflict(preferred: pd.Series, nonpreferred: pd.Series) -> bool:
    present = [column for column in LF_COLS if column in preferred.index]
    if not present:
        return False
    return bool(
        any(preferred[column] == -1 for column in present)
        and any(nonpreferred[column] == 1 for column in present)
    )


def build_pair_table(
    df: pd.DataFrame,
    score_col: str = "y_prob",
    job_col: str = "job_id",
    candidate_col: str = "cand_id",
    pair_delta: float = 0.02,
    max_pairs_per_job: int = 150,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build unique directed preferences and deterministic sampling diagnostics."""
    required = {score_col, job_col, candidate_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing pair-construction columns: {sorted(missing)}")
    if pair_delta < 0:
        raise ValueError("pair_delta must be non-negative")
    if max_pairs_per_job <= 0:
        raise ValueError("max_pairs_per_job must be positive")

    selected_rows = []
    diagnostics = []
    rng = np.random.RandomState(seed)

    for job_id, raw_group in df.groupby(job_col, sort=True):
        group = raw_group.sort_values(
            [candidate_col, "pair_id"] if "pair_id" in raw_group else [candidate_col],
            kind="mergesort",
        ).reset_index(drop=False).rename(columns={"index": "source_index"})
        candidate_pairs = len(group) * (len(group) - 1) // 2
        near_ties = 0
        eligible = []
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                left_score = float(group.at[left, score_col])
                right_score = float(group.at[right, score_col])
                delta = left_score - right_score
                absolute_delta = abs(delta)
                if np.isclose(absolute_delta, 0.0) or (
                    absolute_delta < pair_delta
                    and not np.isclose(absolute_delta, pair_delta)
                ):
                    near_ties += 1
                    continue
                preferred_index, nonpreferred_index = (
                    (left, right) if delta > 0 else (right, left)
                )
                if np.isclose(absolute_delta, pair_delta):
                    absolute_delta = float(pair_delta)
                preferred = group.loc[preferred_index]
                nonpreferred = group.loc[nonpreferred_index]
                row = {
                    "job_id": job_id,
                    "preferred_source_index": int(preferred["source_index"]),
                    "nonpreferred_source_index": int(nonpreferred["source_index"]),
                    "preferred_row_id": _row_identifier(
                        preferred, int(preferred["source_index"])
                    ),
                    "nonpreferred_row_id": _row_identifier(
                        nonpreferred, int(nonpreferred["source_index"])
                    ),
                    "preferred_cand_id": preferred[candidate_col],
                    "nonpreferred_cand_id": nonpreferred[candidate_col],
                    "preferred_y_prob": float(preferred[score_col]),
                    "nonpreferred_y_prob": float(nonpreferred[score_col]),
                    "y_prob_delta": absolute_delta,
                    "hard_negative": _has_lf_conflict(preferred, nonpreferred),
                }
                eligible.append(row)

        hard = [row for row in eligible if row["hard_negative"]]
        ordinary = [row for row in eligible if not row["hard_negative"]]
        if len(hard) >= max_pairs_per_job:
            chosen = [hard[index] for index in rng.choice(
                len(hard), size=max_pairs_per_job, replace=False
            )]
        else:
            remaining = max_pairs_per_job - len(hard)
            if len(ordinary) > remaining:
                ordinary = [ordinary[index] for index in rng.choice(
                    len(ordinary), size=remaining, replace=False
                )]
            chosen = hard + ordinary
        chosen.sort(
            key=lambda row: (
                not row["hard_negative"],
                str(row["preferred_cand_id"]),
                str(row["nonpreferred_cand_id"]),
            )
        )
        selected_rows.extend(chosen)
        diagnostics.append(
            {
                "job_id": job_id,
                "candidate_unordered_pairs": candidate_pairs,
                "near_ties_removed": near_ties,
                "eligible_pairs": len(eligible),
                "eligible_hard_negatives": len(hard),
                "selected_pairs": len(chosen),
                "selected_hard_negatives": sum(
                    row["hard_negative"] for row in chosen
                ),
            }
        )

    pair_table = pd.DataFrame(selected_rows)
    if pair_table.empty:
        pair_table = pd.DataFrame(
            columns=PAIR_ID_COLS
            + [
                "preferred_source_index",
                "nonpreferred_source_index",
                "preferred_y_prob",
                "nonpreferred_y_prob",
                "y_prob_delta",
                "hard_negative",
            ]
        )
    else:
        below_delta = pair_table["y_prob_delta"] < pair_delta
        at_boundary = np.isclose(pair_table["y_prob_delta"], pair_delta)
        if (below_delta & ~at_boundary).any():
            raise AssertionError("Pair below fixed pair_delta")
        if not (
            pair_table["preferred_y_prob"] > pair_table["nonpreferred_y_prob"]
        ).all():
            raise AssertionError("Pair preference direction is invalid")
        canonical = pair_table.apply(
            lambda row: (
                row["job_id"],
                *sorted([row["preferred_row_id"], row["nonpreferred_row_id"]]),
            ),
            axis=1,
        )
        if canonical.duplicated().any():
            raise AssertionError("Duplicate unordered pair")
        if (pair_table.groupby("job_id").size() > max_pairs_per_job).any():
            raise AssertionError("Per-job pair cap exceeded")
    return pair_table.reset_index(drop=True), pd.DataFrame(diagnostics)


def pair_table_to_arrays(
    df: pd.DataFrame,
    pair_table: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
) -> tuple[np.ndarray, np.ndarray]:
    if pair_table.empty:
        return (
            np.zeros((0, len(feature_cols)), dtype=float),
            np.zeros((0, len(feature_cols)), dtype=float),
        )
    preferred = df.loc[
        pair_table["preferred_source_index"].astype(int), feature_cols
    ].to_numpy(float)
    nonpreferred = df.loc[
        pair_table["nonpreferred_source_index"].astype(int), feature_cols
    ].to_numpy(float)
    return preferred, nonpreferred


def pair_table_hash(pair_table: pd.DataFrame) -> str:
    stable = pair_table[PAIR_ID_COLS].sort_values(PAIR_ID_COLS, kind="mergesort")
    payload = stable.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_rank_pairs(
    df: pd.DataFrame,
    score_col: str = "y_prob",
    job_col: str = "job_id",
    max_pairs_per_job: int = 100,
    margin: float = 0.05,
    rng: Optional[np.random.RandomState] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper around the fixed-delta auditable pair builder."""
    rng = rng or np.random.RandomState(42)
    seed = int(rng.randint(0, 2**31 - 1))
    candidate_col = "cand_id" if "cand_id" in df else "pair_id"
    frame = df.copy()
    if candidate_col not in frame:
        frame[candidate_col] = np.arange(len(frame))
    table, _ = build_pair_table(
        frame,
        score_col=score_col,
        job_col=job_col,
        candidate_col=candidate_col,
        pair_delta=margin,
        max_pairs_per_job=max_pairs_per_job,
        seed=seed,
    )
    if all(column in frame for column in FEATURE_COLS):
        return pair_table_to_arrays(frame, table)
    left = table["preferred_y_prob"].to_numpy(float).reshape(-1, 1)
    right = table["nonpreferred_y_prob"].to_numpy(float).reshape(-1, 1)
    return left, right


def build_rank_pairs_robust(
    df: pd.DataFrame,
    score_col: str = "y_prob",
    job_col: str = "job_id",
    max_pairs_per_job: int = 100,
    margin: float = 0.02,
    margin_fallbacks: tuple[float, ...] = (),
    min_pairs: int = 8,
    rng: Optional[np.random.RandomState] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compatibility API; authoritative runs never lower the fixed delta."""
    del margin_fallbacks
    left, right = build_rank_pairs(
        df,
        score_col=score_col,
        job_col=job_col,
        max_pairs_per_job=max_pairs_per_job,
        margin=margin,
        rng=rng,
    )
    if len(left) < min_pairs:
        raise ValueError(
            f"Only {len(left)} pairs at fixed delta {margin}; protocol forbids fallback"
        )
    return left, right, margin


def build_hard_rank_pairs(
    df: pd.DataFrame,
    score_col: str = "y_prob",
    job_col: str = "job_id",
    min_margin: float = 0.02,
    max_margin: float = 1.0,
    max_pairs_per_job: int = 100,
    rng=None,
):
    """Compatibility wrapper using LF-conflict hard-negative priority."""
    del max_margin
    return build_rank_pairs(
        df,
        score_col=score_col,
        job_col=job_col,
        max_pairs_per_job=max_pairs_per_job,
        margin=min_margin,
        rng=rng,
    )
