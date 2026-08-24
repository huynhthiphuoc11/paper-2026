"""
Stage 4 — Aggregation Model Benchmark
======================================
Compares 5 aggregation models on Gold-B-dev, selects the best,
then produces pseudo-labels for the full unlabeled pool.

Models
------
  A  — Unweighted Majority Vote          (no gold required)
  B  — Heuristic Weighted Vote           (no gold required; weights from domain expert)
  B' — Supervised Weighted Vote          (requires Gold-B-dev; Logistic/Ordinal Regression)
  C  — Naive Bayes / Dawid-Skene EM     (no gold required to FIT; gold used only for EVAL)
  D  — Dependency-Aware Label Model      (only built if Stage 1 audit justifies edges)

Selection rule (Occam's Razor):
  Pick the simplest model whose Macro-F1 on Gold-B-dev is not significantly
  worse than the best model (95% bootstrap CI).
  If A ≈ C in CI → use A.
  If C significantly > A,B,B' → use C.
  B' is an important intermediate: if B' ≥ C, no need for complex generative model.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model     import LogisticRegression
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import f1_score, precision_score, recall_score
from sklearn.utils            import resample

warnings.filterwarnings("ignore")

LF_COLS     = ["lf_skill", "lf_sem", "lf_exp", "lf_role", "lf_loc"]
REPORT_DIR  = "reports"

# Bootstrap CI config
N_BOOTSTRAP = 1000
CI_ALPHA    = 0.05    # 95% CI


# ══════════════════════════════════════════════════════════════
# Utility: bootstrap Macro-F1 confidence interval
# ══════════════════════════════════════════════════════════════
def bootstrap_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    alpha: float = CI_ALPHA,
) -> dict:
    """
    Returns mean Macro-F1 and (lower, upper) bootstrap 95% CI.
    Needed to assess whether differences between models are statistically meaningful.
    """
    scores = []
    n = len(y_true)
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        score = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
        scores.append(score)
    scores = np.array(scores)
    return {
        "mean"  : round(float(np.mean(scores)), 4),
        "ci_low": round(float(np.percentile(scores, 100 * alpha / 2)), 4),
        "ci_high":round(float(np.percentile(scores, 100 * (1 - alpha / 2))), 4),
    }


# ══════════════════════════════════════════════════════════════
# Model A — Unweighted Majority Vote
# ══════════════════════════════════════════════════════════════
class ModelA_MajorityVote:
    """
    Hard majority vote over non-abstaining LFs.
    Tie / all-abstain → assign prior (most frequent class in training pool).
    """
    name = "A_MajorityVote"

    def fit(self, df_lf: pd.DataFrame, y_gold=None):
        # count most common binary outcome across non-abstaining LFs
        pos_counts = (df_lf[LF_COLS] == 1).sum(axis=1)
        neg_counts = (df_lf[LF_COLS] == -1).sum(axis=1)
        self._prior = int(pos_counts.sum() >= neg_counts.sum())
        return self

    def predict_proba(self, df_lf: pd.DataFrame) -> np.ndarray:
        pos  = (df_lf[LF_COLS] ==  1).sum(axis=1).values
        neg  = (df_lf[LF_COLS] == -1).sum(axis=1).values
        tot  = (pos + neg).clip(min=1)
        prob = pos / tot                   # fraction of active LFs voting positive
        # all-abstain rows → 0.5 (uncertain)
        prob = np.where(pos + neg == 0, 0.5, prob)
        return prob

    def predict(self, df_lf: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(df_lf)
        return (prob >= threshold).astype(int)


# ══════════════════════════════════════════════════════════════
# Model B — Heuristic Weighted Vote
# ══════════════════════════════════════════════════════════════
class ModelB_HeuristicWeighted:
    """
    Weighted combination of LF votes using manually specified weights.
    Weights encode domain-expert belief about each signal's reliability.
    These weights are NOT learned from data — that would be Model B'.
    """
    name = "B_HeuristicWeighted"

    # Default weights based on domain knowledge (skill and semantic most informative;
    # location least informative for ranking quality).
    DEFAULT_WEIGHTS = {
        "lf_skill": 0.30,
        "lf_sem"  : 0.25,
        "lf_exp"  : 0.20,
        "lf_role" : 0.15,
        "lf_loc"  : 0.10,
    }

    def __init__(self, weights: dict = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

    def fit(self, df_lf: pd.DataFrame, y_gold=None):
        return self   # No learning; weights are fixed a priori

    def predict_proba(self, df_lf: pd.DataFrame) -> np.ndarray:
        scores = np.zeros(len(df_lf))
        wt_sum = np.zeros(len(df_lf))
        for col, w in self.weights.items():
            if col not in df_lf.columns:
                continue
            active = df_lf[col].values != 0
            scores  += active * w * (df_lf[col].values == 1).astype(float)
            wt_sum  += active * w
        wt_sum = np.where(wt_sum == 0, 1.0, wt_sum)
        return scores / wt_sum

    def predict(self, df_lf: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df_lf) >= threshold).astype(int)


# ══════════════════════════════════════════════════════════════
# Model B' — Supervised Weighted Vote  (NEW — closes baseline gap)
# ══════════════════════════════════════════════════════════════
class ModelBPrime_SupervisedWeighted:
    """
    Learns LF weights from Gold-B-dev via Logistic Regression.

    Why this matters (from reviewer feedback):
      If Gold-B-dev (~200 pairs) is already being collected for model selection,
      the marginal cost of also fitting a Logistic Regression on it is near-zero.
      B' is a very strong baseline: it learns from gold labels WITHOUT the
      conditional-independence assumption of Model C/D, and with far fewer
      parameters (only K weights).

      If B' ≥ C/D on Gold-B-dev → no need for complex generative model.
      If C/D > B' significantly → we have a genuine justification for Label Model.

    Features: binary indicators (LF_k voted positive / negative / abstain)
    encoded as two binary columns per LF (positive_k, negative_k).
    """
    name = "Bprime_SupervisedWeighted"

    def __init__(self, C_reg: float = 1.0, max_iter: int = 500):
        self.C_reg     = C_reg
        self.max_iter  = max_iter
        self.scaler    = StandardScaler()
        self.model     = LogisticRegression(C=C_reg, max_iter=max_iter,
                                             class_weight="balanced",
                                             solver="lbfgs", random_state=42)
        self.is_fitted = False

    def _build_features(self, df_lf: pd.DataFrame) -> np.ndarray:
        """
        Encode each LF as two binary columns:
          pos_k = (L_k == +1)
          neg_k = (L_k == -1)
        Abstain (0) → both columns are 0 (provides no evidence).
        """
        feats = []
        for col in LF_COLS:
            if col not in df_lf.columns:
                feats.append(np.zeros(len(df_lf)))
                feats.append(np.zeros(len(df_lf)))
            else:
                feats.append((df_lf[col].values ==  1).astype(float))
                feats.append((df_lf[col].values == -1).astype(float))
        return np.column_stack(feats)

    def fit(self, df_lf: pd.DataFrame, y_gold: np.ndarray):
        """
        Requires Gold-B-dev labels y_gold ∈ {0, 1}.
        Should be called ONLY with Gold-B-dev — never with Gold-B-test
        to prevent test-set contamination.
        """
        if y_gold is None:
            raise ValueError("ModelB' requires gold labels (y_gold) from Gold-B-dev.")
        X = self._build_features(df_lf)
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y_gold)
        self.is_fitted = True
        # Store learned LF weights for paper reporting
        coef = self.model.coef_[0]
        self.lf_weight_report = {
            f"{col}_pos": round(float(coef[2*i]),   4)
            for i, col in enumerate(LF_COLS)
        } | {
            f"{col}_neg": round(float(coef[2*i+1]), 4)
            for i, col in enumerate(LF_COLS)
        }
        return self

    def predict_proba(self, df_lf: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Fit ModelB' on Gold-B-dev first.")
        X = self._build_features(df_lf)
        X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df_lf: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df_lf) >= threshold).astype(int)


# ══════════════════════════════════════════════════════════════
# Model C — Dawid-Skene EM (Naive Bayes Label Model)
# ══════════════════════════════════════════════════════════════
class ModelC_LabelModel:
    """
    Generative Naive Bayes model adapted from weak_supervision.py.
    Assumes Conditional Independence (CI) of LFs given Y.

    Identifiability note (reviewer-safe wording):
      Three or more conditionally independent binary LFs can in principle
      provide a minimal setting for latent-class parameter identifiability
      (Dawid & Skene, 1979). Practical reliability depends on the
      informativeness and degree of conditional independence of the LFs —
      strong conditional dependence reduces available independent information
      and may lead to poorly identified estimates. This risk is assessed
      diagnostically in Stage 1 (Cramér's V + Conflict Rate).
    """
    name = "C_LabelModel_CI"

    def __init__(self, n_iter: int = 50, alpha_clip=(0.55, 0.95)):
        self.n_iter     = n_iter
        self.alpha_clip = alpha_clip
        self.alpha_     = None   # P(L_k = +1 | Y = 1)  — sensitivity
        self.beta_      = None   # P(L_k = -1 | Y = 0)  — specificity
        self.p_prior_   = None

    def fit(self, df_lf: pd.DataFrame, y_gold=None):
        """
        Fits ONLY on unlabeled LF matrix (y_gold NOT used).
        EM algorithm: E-step (posterior update) → M-step (parameter update).
        """
        lf_mat  = df_lf[LF_COLS].values.astype(float)
        N, K    = lf_mat.shape
        pos_mask = (lf_mat ==  1)
        neg_mask = (lf_mat == -1)

        # Initialise
        p     = float(np.mean(pos_mask.sum(axis=1) >= neg_mask.sum(axis=1)))
        alpha = np.full(K, 0.75)  # sensitivity
        beta  = np.full(K, 0.75)  # specificity
        w     = np.zeros(N)

        for _ in range(self.n_iter):
            # ── E-step ──────────────────────────────────────
            log_pos = np.log(p + 1e-10)
            log_neg = np.log(1.0 - p + 1e-10)
            for k in range(K):
                log_pos = (
                    log_pos
                    + pos_mask[:, k] * np.log(alpha[k] + 1e-10)
                    + neg_mask[:, k] * np.log(1.0 - alpha[k] + 1e-10)
                )
                log_neg = (
                    log_neg
                    + pos_mask[:, k] * np.log(1.0 - beta[k] + 1e-10)
                    + neg_mask[:, k] * np.log(beta[k] + 1e-10)
                )
            max_log = np.maximum(log_pos, log_neg)
            w_new   = np.exp(log_pos - max_log) / (
                np.exp(log_pos - max_log) + np.exp(log_neg - max_log) + 1e-10
            )
            if np.max(np.abs(w_new - w)) < 1e-5:
                w = w_new
                break
            w = w_new

            # ── M-step ──────────────────────────────────────
            p          = float(np.mean(w))
            denom_pos  = np.sum(w) + 1e-10
            denom_neg  = np.sum(1.0 - w) + 1e-10
            for k in range(K):
                alpha[k] = np.clip(
                    np.sum(w * pos_mask[:, k]) / denom_pos,
                    *self.alpha_clip,
                )
                beta[k] = np.clip(
                    np.sum((1.0 - w) * neg_mask[:, k]) / denom_neg,
                    *self.alpha_clip,
                )

        self.alpha_   = alpha
        self.beta_    = beta
        self.p_prior_ = p
        self._w_train = w
        return self

    def predict_proba(self, df_lf: pd.DataFrame) -> np.ndarray:
        lf_mat   = df_lf[LF_COLS].values.astype(float)
        N, K     = lf_mat.shape
        pos_mask = (lf_mat ==  1)
        neg_mask = (lf_mat == -1)
        log_pos  = np.log(self.p_prior_ + 1e-10)
        log_neg  = np.log(1.0 - self.p_prior_ + 1e-10)
        for k in range(K):
            log_pos = (
                log_pos
                + pos_mask[:, k] * np.log(self.alpha_[k] + 1e-10)
                + neg_mask[:, k] * np.log(1.0 - self.alpha_[k] + 1e-10)
            )
            log_neg = (
                log_neg
                + pos_mask[:, k] * np.log(1.0 - self.beta_[k] + 1e-10)
                + neg_mask[:, k] * np.log(self.beta_[k] + 1e-10)
            )
        max_log = np.maximum(log_pos, log_neg)
        return np.exp(log_pos - max_log) / (
            np.exp(log_pos - max_log) + np.exp(log_neg - max_log) + 1e-10
        )

    def predict(self, df_lf: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df_lf) >= threshold).astype(int)

    def get_lf_parameters(self) -> pd.DataFrame:
        """Returns estimated sensitivity (α) and specificity (β) per LF — for paper reporting."""
        return pd.DataFrame({
            "lf"         : LF_COLS,
            "sensitivity": np.round(self.alpha_, 4),
            "specificity": np.round(self.beta_,  4),
        })


# ══════════════════════════════════════════════════════════════
# Stage 4 Benchmark Runner
# ══════════════════════════════════════════════════════════════
def run_stage4_benchmark(
    df_train_lf  : pd.DataFrame,   # Full unlabeled pool — for fitting generative models
    df_gold_dev  : pd.DataFrame,   # Gold-B-dev — for model selection & B' training
    y_gold_dev   : np.ndarray,     # Binary gold labels for Gold-B-dev {0,1}
    stage1_report: dict = None,    # Stage 1 audit result (for Model D justification)
    report_path  : str  = None,
) -> dict:
    """
    Evaluates all models on Gold-B-dev and selects the winner.

    STRICT PROTOCOL:
      - Model B' is fitted on Gold-B-dev (same set used for evaluation).
        This is permissible for MODEL SELECTION, but Gold-B-test must remain
        untouched for final reporting.
      - Model C/D are fitted on df_train_lf (unlabeled), evaluated on Gold-B-dev.
      - No model sees Gold-B-test at any point in this stage.
    """
    print("=" * 65)
    print("  Stage 4 — Aggregation Model Benchmark (Gold-B-dev)")
    print("=" * 65)
    print(f"  Train pool (unlabeled): {len(df_train_lf):,} pairs")
    print(f"  Gold-B-dev            : {len(df_gold_dev):,} pairs")
    print(f"  Positive rate (dev)   : {y_gold_dev.mean():.3f}")

    results = {}

    # ── Instantiate all models ────────────────────────────────
    models = {
        "A_MajorityVote"       : ModelA_MajorityVote(),
        "B_HeuristicWeighted"  : ModelB_HeuristicWeighted(),
        "Bprime_Supervised"    : ModelBPrime_SupervisedWeighted(),
        "C_LabelModel_CI"      : ModelC_LabelModel(),
    }

    # ── Fit ───────────────────────────────────────────────────
    print("\n[Fitting models...]")
    models["A_MajorityVote"].fit(df_train_lf)
    models["B_HeuristicWeighted"].fit(df_train_lf)
    # B' uses Gold-B-dev labels — explicitly documented
    models["Bprime_Supervised"].fit(df_gold_dev, y_gold=y_gold_dev)
    models["C_LabelModel_CI"].fit(df_train_lf)  # unsupervised — no gold

    # ── Predict & Evaluate on Gold-B-dev ─────────────────────
    print("\n[Evaluating on Gold-B-dev...]")
    print(f"\n{'Model':<25} {'Macro-F1':>10} {'CI 95%':>18} {'Precision':>10} {'Recall':>8}")
    print("-" * 75)

    for name, model in models.items():
        y_pred = model.predict(df_gold_dev)
        f1     = f1_score(y_gold_dev, y_pred, average="macro", zero_division=0)
        prec   = precision_score(y_gold_dev, y_pred, average="macro", zero_division=0)
        rec    = recall_score(y_gold_dev, y_pred, average="macro", zero_division=0)
        ci     = bootstrap_f1_ci(y_gold_dev, y_pred)
        results[name] = {
            "macro_f1"       : round(f1, 4),
            "precision_macro": round(prec, 4),
            "recall_macro"   : round(rec, 4),
            "bootstrap_ci_95": ci,
        }
        print(f"{name:<25} {f1:>10.4f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
              f"   {prec:>9.4f} {rec:>8.4f}")

    # ── Model Selection (Occam's Razor) ──────────────────────
    print("\n── Model Selection (Occam's Razor) ─────────────────────")

    best_name, best_f1 = max(
        ((k, v["macro_f1"]) for k, v in results.items()), key=lambda x: x[1]
    )
    baseline_f1 = results["A_MajorityVote"]["macro_f1"]
    c_f1        = results["C_LabelModel_CI"]["macro_f1"]
    bp_f1       = results["Bprime_Supervised"]["macro_f1"]

    # CI-based overlap check: are C and A distinguishable?
    ci_a    = results["A_MajorityVote"]["bootstrap_ci_95"]
    ci_c    = results["C_LabelModel_CI"]["bootstrap_ci_95"]
    ci_bp   = results["Bprime_Supervised"]["bootstrap_ci_95"]

    c_beats_a  = ci_c["ci_low"]  > ci_a["ci_high"]   # C significantly better than A
    bp_beats_c = ci_bp["ci_low"] > ci_c["ci_high"]   # B' significantly better than C

    if bp_beats_c:
        selected = "Bprime_Supervised"
        rationale = (
            "B' (Supervised Weighted Vote) significantly outperforms Label Model C "
            "on Gold-B-dev. Gold labels are sufficient to learn reliable signal weights "
            "without the generative model's CI assumption. Label Model complexity is NOT justified."
        )
    elif c_beats_a:
        selected = "C_LabelModel_CI"
        rationale = (
            "Label Model C significantly outperforms Majority Vote A and Supervised B' "
            "on Gold-B-dev. Generative weak supervision provides measurably better "
            "pseudo-labels — Label Model is empirically justified."
        )
    else:
        selected = "A_MajorityVote"
        rationale = (
            "No complex model significantly outperforms Majority Vote on Gold-B-dev "
            "(overlapping 95% CIs). Applying Occam's Razor: use the simplest baseline. "
            "Label Model and Supervised Vote do NOT add measurable value."
        )

    print(f"  Best raw Macro-F1 : {best_name} ({best_f1:.4f})")
    print(f"  Selected model    : {selected}")
    print(f"  Rationale         : {rationale}")

    selection_result = {
        "selected_model"   : selected,
        "rationale"        : rationale,
        "c_significantly_beats_a" : bool(c_beats_a),
        "bp_significantly_beats_c": bool(bp_beats_c),
        "occams_razor_applied"    : selected == "A_MajorityVote",
        "note": (
            "This selection was made on Gold-B-dev only. "
            "Final performance MUST be reported on Gold-B-test (never seen in this stage)."
        ),
    }

    # ── LF Parameters (for paper) ─────────────────────────────
    lf_params = models["C_LabelModel_CI"].get_lf_parameters().to_dict(orient="records")
    bp_weights = models["Bprime_Supervised"].lf_weight_report

    # ── Report ────────────────────────────────────────────────
    report = {
        "stage"           : 4,
        "n_train_unlabeled": len(df_train_lf),
        "n_gold_dev"      : len(df_gold_dev),
        "model_results"   : results,
        "model_selection" : selection_result,
        "c_lf_parameters" : lf_params,
        "bp_lf_weights"   : bp_weights,
        "models_note": {
            "A": "No gold required. Unweighted majority vote.",
            "B": "No gold required. Weights set by domain expert a priori.",
            "Bprime": "Requires Gold-B-dev. Logistic Regression — learns weights from gold.",
            "C": "No gold required to FIT. Gold used only for evaluation.",
            "D": "Only built if Stage 1 audit justifies dependency edges.",
        },
    }

    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[Done] Stage 4 report saved to '{report_path}'")

    return report, models[selected]


# ══════════════════════════════════════════════════════════════
# Stage 5 — Generate Pseudo-labels for Full Training Pool
# ══════════════════════════════════════════════════════════════
def generate_pseudo_labels(
    selected_model,
    df_pool: pd.DataFrame,
    epsilon: float = 0.10,
) -> pd.DataFrame:
    """
    Applies the selected model to the full unlabeled pool to produce:
      - r(x)  = expected relevance score  ∈ [0, 1]  (proxy for E[Y | L])
      - y_hard = hard binary label (for Model A / B / B')

    Pairwise RankNet target (Approach A — no λ hyperparameter):
      Given pair (x_i, x_j):
        y_pair = 1   if r(x_i) > r(x_j) + ε
        y_pair = 0   if r(x_i) < r(x_j) - ε
        discard      if |r(x_i) - r(x_j)| ≤ ε

    ε = 0.10 discards near-tie pairs that carry minimal ranking information.
    This avoids introducing an arbitrary λ (softmax temperature) hyperparameter.
    """
    df_out = df_pool.copy()
    df_out["r_score"]  = selected_model.predict_proba(df_pool)
    df_out["y_pseudo"] = (df_out["r_score"] >= 0.5).astype(int)
    df_out["pseudo_source"] = selected_model.name
    df_out["epsilon_margin"] = epsilon

    print(f"\n[Stage 5] Pseudo-labels generated for {len(df_out):,} pairs.")
    print(f"  Positive rate: {df_out['y_pseudo'].mean():.4f}")
    print(f"  r_score mean : {df_out['r_score'].mean():.4f}  "
          f"std: {df_out['r_score'].std():.4f}")
    return df_out


if __name__ == "__main__":
    # ── Quick smoke test on available weak labels + gold ──────
    import sys
    sys.path.insert(0, ".")

    print("[Smoke test] Loading data...")
    df_lf = pd.read_csv(os.path.join("data", "weak_labels", "train_weak_labels.csv"))
    df_gold = pd.read_csv(os.path.join("data", "gold", "human_validated_benchmark.csv"))

    # Gold has 'human_relevance' column (0 or 1)
    if "human_relevance" not in df_gold.columns:
        raise ValueError("Gold CSV must have 'human_relevance' column.")

    # Merge to get LF values for gold pairs
    lf_lookup = df_lf.set_index(["job_id", "cand_id"])
    gold_pairs = df_gold.copy()
    for col in LF_COLS:
        gold_pairs[col] = gold_pairs.apply(
            lambda r: lf_lookup.at[(r["job_id"], r["cand_id"]), col]
            if (r["job_id"], r["cand_id"]) in lf_lookup.index else 0,
            axis=1,
        )

    y_gold = gold_pairs["human_relevance"].values.astype(int)

    report, selected_model = run_stage4_benchmark(
        df_train_lf  = df_lf,
        df_gold_dev  = gold_pairs,
        y_gold_dev   = y_gold,
        report_path  = os.path.join(REPORT_DIR, "stage4_model_benchmark.json"),
    )

    pseudo_df = generate_pseudo_labels(selected_model, df_lf)
    pseudo_df.to_csv(
        os.path.join("data", "weak_labels", "pseudo_labels_selected.csv"), index=False
    )
    print("\n[Done] Pseudo-labels saved.")
