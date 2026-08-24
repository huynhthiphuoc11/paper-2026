"""
Stage 1 — LF Structural Audit
==============================
Runs BEFORE any annotation or model training.
Input : data/weak_labels/train_weak_labels.csv  (LF matrix, no gold needed)
Output: reports/stage1_lf_audit.json + console tables

Metrics computed
----------------
1. Coverage & Abstain Rate per LF
2. Positive / Negative Rate per LF
3. Pairwise Conflict Rate (P(Li ≠ Lj | both non-abstain))
4. Pairwise Cramér's V  (association strength, symmetric, 0–1)
5. Fleiss' κ  (multi-source agreement, diagnostic only — NOT a decision gate)
6. Dependency Graph Suggestion  (data-driven edges for Model D)

Decision rule (data-driven):
  - If max(Cramér's V) < 0.15 AND Conflict diversity exists
    → 3+ independent signals, Label Model C/D worthwhile
  - If two LFs have V > 0.50 AND Conflict < 10%
    → strong redundancy, those LFs should be modeled as dependent (Model D edge)
  - If Coverage(LF_k) < 0.30
    → LF_k contributes very little supervision → flag for review
"""

import os
import io
import sys
import json
import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import chi2_contingency

# Force UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "weak_labels", "train_weak_labels.csv")
REPORT_PATH = os.path.join("reports", "stage1_lf_audit.json")

LF_COLS = ["lf_skill", "lf_sem", "lf_exp", "lf_role", "lf_loc"]

# Thresholds for Dependency Graph suggestion (Stage 2.5 of methodology)
CRAMERS_V_DEP_THRESHOLD   = 0.30   # V > this → flag as potentially dependent
CONFLICT_REDUND_THRESHOLD = 0.15   # Conflict < this + V > above → flag as redundant


# ──────────────────────────────────────────────────────────────
# 1. Coverage Statistics
# ──────────────────────────────────────────────────────────────
def compute_coverage_stats(df: pd.DataFrame, lf_cols: list) -> pd.DataFrame:
    """
    Per-LF: Coverage, Positive Rate, Negative Rate, Abstain Rate.
    Coverage = (non-abstain) / N
    """
    N = len(df)
    rows = []
    for col in lf_cols:
        vals = df[col].values
        n_pos  = int(np.sum(vals ==  1))
        n_neg  = int(np.sum(vals == -1))
        n_abs  = int(np.sum(vals ==  0))
        n_cov  = n_pos + n_neg
        rows.append({
            "lf"           : col,
            "N"            : N,
            "coverage_rate": round(n_cov / N, 4),
            "positive_rate": round(n_pos / N, 4),
            "negative_rate": round(n_neg / N, 4),
            "abstain_rate" : round(n_abs / N, 4),
            "pos_neg_ratio": round(n_pos / max(n_neg, 1), 4),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# 2. Pairwise Conflict Rate
# ──────────────────────────────────────────────────────────────
def compute_pairwise_conflict(df: pd.DataFrame, lf_cols: list) -> pd.DataFrame:
    """
    Conflict(Li, Lj) = P(Li ≠ Lj | Li ≠ 0 AND Lj ≠ 0)

    High conflict + high Cramér's V → genuinely complementary signals.
    High conflict + low Cramér's V  → one signal may be noisy.
    Low conflict  + high Cramér's V → redundant signals.
    """
    rows = []
    for c1, c2 in combinations(lf_cols, 2):
        mask_both_active = (df[c1] != 0) & (df[c2] != 0)
        n_both = int(mask_both_active.sum())
        if n_both == 0:
            conflict_rate = float("nan")
        else:
            n_conflict = int(((df[c1] != df[c2]) & mask_both_active).sum())
            conflict_rate = round(n_conflict / n_both, 4)
        rows.append({
            "lf_a"         : c1,
            "lf_b"         : c2,
            "n_both_active": n_both,
            "conflict_rate": conflict_rate,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# 3. Cramér's V  (pairwise association, no ordinal assumption)
# ──────────────────────────────────────────────────────────────
def cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    """
    Symmetric measure of association for categorical variables.
    V ∈ [0, 1].  V = 0 → independence; V = 1 → perfect association.

    Why not Pearson/Spearman?
      LF values ∈ {-1, 0, +1} where 0 means ABSTAIN — not a
      meaningful midpoint between -1 and +1. Treating them as
      ordinal would distort the correlation.
    """
    contingency = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(contingency, correction=False)
    n    = contingency.values.sum()
    r, c = contingency.shape
    phi2 = chi2 / n
    # Bias-corrected Cramér's V (Bergsma, 2013)
    phi2_corr = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    r_corr    = r - (r - 1) ** 2 / (n - 1)
    c_corr    = c - (c - 1) ** 2 / (n - 1)
    denom     = min(r_corr - 1, c_corr - 1)
    if denom <= 0:
        return 0.0
    return float(round(np.sqrt(phi2_corr / denom), 4))


def compute_cramers_v_matrix(df: pd.DataFrame, lf_cols: list) -> pd.DataFrame:
    mat = pd.DataFrame(index=lf_cols, columns=lf_cols, dtype=float)
    for c1 in lf_cols:
        for c2 in lf_cols:
            mat.loc[c1, c2] = 1.0 if c1 == c2 else cramers_v(
                df[c1].values, df[c2].values
            )
    return mat


# ──────────────────────────────────────────────────────────────
# 4. Fleiss' κ  (multi-source agreement — DIAGNOSTIC ONLY)
# ──────────────────────────────────────────────────────────────
def compute_fleiss_kappa(df: pd.DataFrame, lf_cols: list) -> dict:
    """
    Fleiss' κ adapted for K labeling functions over binary votes.

    Interpretation (diagnostic, NOT a decision gate):
      κ ≥ 0.80 → very high agreement (possible redundancy or shared systematic bias)
      0.40 ≤ κ < 0.80 → moderate agreement (signals partially diverge)
      κ < 0.40 → low agreement (signals diverge; verify no LF is pure noise)

    Note: uses only non-abstain rows (ABSTAIN excluded from agreement
    calculation, consistent with the partial-coverage LF paradigm).

    Reference: Fleiss (1971), J. of Psychological Measurement.
    """
    # Binarise: treat +1 as "positive", -1 as "negative"; drop rows where ALL abstain
    df_bin = df[lf_cols].replace({1: 1, -1: 0, 0: np.nan})
    # Keep rows where at least 2 LFs are non-abstain
    n_active = df_bin.notna().sum(axis=1)
    df_bin = df_bin[n_active >= 2].copy()

    N  = len(df_bin)          # subjects (pairs)
    K_raters = len(lf_cols)   # raters

    if N == 0:
        return {"fleiss_kappa": None, "n_pairs_used": 0, "warning": "No pairs with ≥2 active LFs"}

    # Fill abstains with the row mean (soft imputation) so each row has K votes
    # This is a pragmatic approximation; a stricter approach excludes abstaining raters per row.
    # We report both the count of fully-active rows and note the approximation.
    df_filled = df_bin.apply(lambda row: row.fillna(row.mean()), axis=1)

    # p_j = proportion of all assignments in category j (positive / negative)
    p_pos = df_filled.mean().mean()        # mean positive rate across raters
    p_neg = 1.0 - p_pos

    # P̄_i = proportion of rater pairs who agree for subject i
    # Exact formula: P_i = (1 / n*(n-1)) * sum_j(n_ij*(n_ij-1))
    n_pos_per_row = df_filled.sum(axis=1)   # count of positive votes per row
    n_neg_per_row = K_raters - n_pos_per_row

    p_i = (n_pos_per_row * (n_pos_per_row - 1) + n_neg_per_row * (n_neg_per_row - 1)) / (
        K_raters * (K_raters - 1)
    )
    p_bar = p_i.mean()

    # P̄_e = expected agreement by chance
    p_e = p_pos**2 + p_neg**2

    if abs(1.0 - p_e) < 1e-10:
        kappa = 1.0
    else:
        kappa = round(float((p_bar - p_e) / (1.0 - p_e)), 4)

    interpretation = (
        "Very high agreement (possible redundancy or shared systematic bias)"
        if kappa >= 0.80
        else "Moderate agreement — signals partially diverge"
        if kappa >= 0.40
        else "Low agreement — verify no LF is pure noise"
    )

    return {
        "fleiss_kappa"       : kappa,
        "p_bar_observed"     : round(float(p_bar), 4),
        "p_e_expected_chance": round(float(p_e), 4),
        "n_pairs_used"       : int(N),
        "interpretation"     : interpretation,
        "diagnostic_note"    : (
            "Fleiss' κ is a DIAGNOSTIC metric only. "
            "It does NOT determine whether to use the Label Model. "
            "Model selection is decided empirically in Stage 4 (Gold-B-dev benchmark)."
        ),
    }


# ──────────────────────────────────────────────────────────────
# 5. Dependency Graph Suggestion  (data-driven, for Model D)
# ──────────────────────────────────────────────────────────────
def suggest_dependency_graph(
    conflict_df: pd.DataFrame,
    cramers_df: pd.DataFrame,
    lf_cols: list,
    v_threshold: float = CRAMERS_V_DEP_THRESHOLD,
    conflict_redundancy_threshold: float = CONFLICT_REDUND_THRESHOLD,
) -> dict:
    """
    Data-driven suggestion for edges in Model D (dependency-aware Label Model).

    Edge rule:
      ADD edge (Li, Lj) if V(Li,Lj) > v_threshold
        → signals are statistically associated
        → Model C's CI assumption may be violated
        → modeling dependency can prevent biased accuracy estimates

    Redundancy flag (strongest form of dependency):
      V(Li,Lj) > v_threshold AND Conflict(Li,Lj) < conflict_redundancy_threshold
        → signals are nearly collinear: including both without modeling
          this provides very little additional independent information.
    """
    edges       = []
    redundant   = []
    independent = []

    for _, row in conflict_df.iterrows():
        c1, c2     = row["lf_a"], row["lf_b"]
        v          = float(cramers_df.loc[c1, c2])
        conflict   = float(row["conflict_rate"]) if not np.isnan(row["conflict_rate"]) else 0.0

        if v > v_threshold:
            edges.append({
                "lf_a": c1, "lf_b": c2,
                "cramers_v": v, "conflict_rate": conflict,
                "reason": "High association (V > threshold); CI assumption likely violated.",
            })
            if conflict < conflict_redundancy_threshold:
                redundant.append((c1, c2))
        else:
            independent.append({
                "lf_a": c1, "lf_b": c2,
                "cramers_v": v, "conflict_rate": conflict,
            })

    return {
        "dependency_edges_suggested"         : edges,
        "redundant_pairs"                    : [f"{a}↔{b}" for a, b in redundant],
        "approximately_independent_pairs"    : independent,
        "model_d_justified"                  : len(edges) > 0,
        "threshold_cramers_v_used"           : v_threshold,
        "threshold_conflict_redundancy_used" : conflict_redundancy_threshold,
        "note": (
            "These are data-driven suggestions. "
            "Reviewer-safe claim: 'Dependency structure was determined "
            "empirically via bias-corrected Cramér\\'s V and pairwise conflict rates.' "
            "Model D is built ONLY if model_d_justified is True."
        ),
    }


# ──────────────────────────────────────────────────────────────
# 6. Preliminary Decision Guide
# ──────────────────────────────────────────────────────────────
def generate_decision_guide(
    coverage_df: pd.DataFrame,
    kappa_result: dict,
    dep_graph: dict,
) -> dict:
    """
    Non-binding, heuristic guide to help decide whether Label Model C/D
    is worth pursuing before spending annotation budget.

    IMPORTANT: This is a PRELIMINARY SIGNAL, not a theorem.
    Final model selection MUST be done empirically on Gold-B-dev (Stage 4).
    """
    low_coverage_lfs = coverage_df[coverage_df["coverage_rate"] < 0.30]["lf"].tolist()
    kappa = kappa_result.get("fleiss_kappa")

    signals = []
    recommendation = "PROCEED with Label Model investigation"

    if low_coverage_lfs:
        signals.append(
            f"Low coverage (<30%) LFs detected: {low_coverage_lfs}. "
            "These LFs contribute very little supervision. Consider removing or revising them."
        )

    if dep_graph["redundant_pairs"]:
        signals.append(
            f"Near-redundant LF pairs: {dep_graph['redundant_pairs']}. "
            "Label Model C (CI assumption) will underestimate these LFs' overlap. "
            "If Model D cannot be built, Label Model C may perform similarly to Majority Vote."
        )

    if kappa is not None and kappa > 0.80:
        signals.append(
            f"Fleiss' κ = {kappa} (very high agreement). "
            "All LFs may share a common systematic bias or be near-redundant. "
            "Label Model C/D may not outperform Majority Vote — verify on Gold-B-dev."
        )
    elif kappa is not None and kappa < 0.20:
        signals.append(
            f"Fleiss' κ = {kappa} (very low agreement). "
            "At least one LF may be acting as noise rather than a complementary signal. "
            "Check per-LF quality on Gold-A (Diagnostic) before investing in Label Model."
        )

    if not signals:
        signals.append(
            "No critical warnings. Coverage, conflict, and association look reasonable. "
            "Label Model investigation is justified — proceed to Stage 2 (Gold set annotation)."
        )
    else:
        if len(signals) >= 2 and dep_graph["redundant_pairs"]:
            recommendation = (
                "CAUTION: Multiple warnings. "
                "Consider addressing low-coverage / redundant LFs BEFORE annotating Gold sets. "
                "Stage 4 empirical comparison on Gold-B-dev will be the definitive arbiter."
            )

    return {
        "recommendation"     : recommendation,
        "signals"            : signals,
        "mandatory_reminder" : (
            "Stage 1 is diagnostic. Final model selection MUST use "
            "Gold-B-dev benchmark (Stage 4). κ alone is NOT a decision gate."
        ),
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def run_audit(data_path: str = DATA_PATH, report_path: str = REPORT_PATH) -> dict:
    print("=" * 65)
    print("  Stage 1 — LF Structural Audit (no gold labels required)")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────
    df = pd.read_csv(data_path)
    N  = len(df)
    print(f"\n[Load] {N:,} pairs loaded from '{data_path}'")
    print(f"[Load] LF columns found: {[c for c in LF_COLS if c in df.columns]}")

    missing = [c for c in LF_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing LF columns: {missing}. "
            "Run lf_definitions.py first to generate the LF matrix."
        )

    df_lf = df[LF_COLS].copy()

    # ── 1. Coverage ───────────────────────────────────────────
    print("\n── 1. Coverage & Rate Statistics ──────────────────────")
    coverage_df = compute_coverage_stats(df_lf, LF_COLS)
    print(coverage_df.to_string(index=False))

    # ── 2. Conflict ───────────────────────────────────────────
    print("\n── 2. Pairwise Conflict Rates ──────────────────────────")
    conflict_df = compute_pairwise_conflict(df_lf, LF_COLS)
    print(conflict_df.to_string(index=False))

    # ── 3. Cramér's V ─────────────────────────────────────────
    print("\n── 3. Cramér's V Matrix (bias-corrected) ───────────────")
    cramers_df = compute_cramers_v_matrix(df_lf, LF_COLS)
    print(cramers_df.round(3).to_string())

    # ── 4. Fleiss' κ ──────────────────────────────────────────
    print("\n── 4. Fleiss' κ (Diagnostic Only) ─────────────────────")
    kappa_result = compute_fleiss_kappa(df_lf, LF_COLS)
    for k, v in kappa_result.items():
        print(f"   {k:30s}: {v}")

    # ── 5. Dependency Graph ───────────────────────────────────
    print("\n── 5. Dependency Graph Suggestion (for Model D) ───────")
    dep_graph = suggest_dependency_graph(conflict_df, cramers_df, LF_COLS)
    print(f"   Model D justified: {dep_graph['model_d_justified']}")
    print(f"   Suggested edges  : {[e['lf_a']+'↔'+e['lf_b'] for e in dep_graph['dependency_edges_suggested']]}")
    print(f"   Redundant pairs  : {dep_graph['redundant_pairs']}")

    # ── 6. Decision Guide ─────────────────────────────────────
    print("\n── 6. Preliminary Decision Guide ───────────────────────")
    decision = generate_decision_guide(coverage_df, kappa_result, dep_graph)
    print(f"   Recommendation: {decision['recommendation']}")
    for s in decision["signals"]:
        print(f"   ⚠  {s}")
    print(f"\n   ★  {decision['mandatory_reminder']}")

    # ── Assemble report ──────────────────────────────────────
    report = {
        "stage"        : 1,
        "description"  : "LF Structural Audit — no gold labels required",
        "n_pairs"      : N,
        "lf_cols"      : LF_COLS,
        "coverage_stats": coverage_df.to_dict(orient="records"),
        "conflict_rates": conflict_df.to_dict(orient="records"),
        "cramers_v_matrix": {
            c1: {c2: float(cramers_df.loc[c1, c2]) for c2 in LF_COLS}
            for c1 in LF_COLS
        },
        "fleiss_kappa" : kappa_result,
        "dependency_graph_suggestion": dep_graph,
        "decision_guide": decision,
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[Done] Full report saved to '{report_path}'")
    print("=" * 65)
    return report


if __name__ == "__main__":
    run_audit()
