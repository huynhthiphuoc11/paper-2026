from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEFAULT_LF_COLS = ["lf_skill", "lf_sem", "lf_exp", "lf_role", "lf_loc"]


class ProbabilisticLabelAggregator:
    """Aggregate {-1, 0, +1} labeling functions into a weak probability."""

    def __init__(self, method="dawid_skene", n_iter=30, parameter_clip=(0.55, 0.95)):
        self.method = method
        self.n_iter = n_iter
        self.parameter_clip = parameter_clip
        self.source_sensitivities = None
        self.source_specificities = None
        self.p_prior = 0.50
        self.lf_cols = None
        self.is_fitted = False

    @staticmethod
    def _posterior(lf_matrix, p, alpha, beta):
        pos_mask = lf_matrix == 1
        neg_mask = lf_matrix == -1
        log_pos = np.full(len(lf_matrix), np.log(p + 1e-10), dtype=float)
        log_neg = np.full(len(lf_matrix), np.log(1.0 - p + 1e-10), dtype=float)

        for k in range(lf_matrix.shape[1]):
            log_pos += pos_mask[:, k] * np.log(alpha[k] + 1e-10)
            log_pos += neg_mask[:, k] * np.log(1.0 - alpha[k] + 1e-10)
            log_neg += pos_mask[:, k] * np.log(1.0 - beta[k] + 1e-10)
            log_neg += neg_mask[:, k] * np.log(beta[k] + 1e-10)

        max_log = np.maximum(log_pos, log_neg)
        pos_exp = np.exp(log_pos - max_log)
        neg_exp = np.exp(log_neg - max_log)
        return pos_exp / (pos_exp + neg_exp + 1e-10)

    def fit(self, df_lfs, lf_cols=DEFAULT_LF_COLS):
        self.lf_cols = list(lf_cols)
        lf_matrix = df_lfs[self.lf_cols].to_numpy(dtype=int)
        if len(lf_matrix) == 0:
            raise ValueError("Cannot fit label aggregator on an empty dataframe")

        if self.method == "consensus":
            self.is_fitted = True
            return self
        if self.method != "dawid_skene":
            raise ValueError(f"Unknown aggregation method: {self.method}")

        pos_mask = lf_matrix == 1
        neg_mask = lf_matrix == -1
        active_rows = (pos_mask | neg_mask).any(axis=1)
        if active_rows.any():
            p = float(
                np.mean(
                    pos_mask[active_rows].sum(axis=1)
                    >= neg_mask[active_rows].sum(axis=1)
                )
            )
        else:
            p = 0.50
        p = float(np.clip(p, 0.05, 0.95))

        n_lfs = lf_matrix.shape[1]
        alpha = np.full(n_lfs, 0.75, dtype=float)
        beta = np.full(n_lfs, 0.75, dtype=float)
        previous = None

        for _ in range(self.n_iter):
            weights = self._posterior(lf_matrix, p, alpha, beta)
            p = float(np.clip(np.mean(weights), 0.05, 0.95))

            for k in range(n_lfs):
                active = pos_mask[:, k] | neg_mask[:, k]
                denom_pos = np.sum(weights[active])
                denom_neg = np.sum(1.0 - weights[active])
                if denom_pos > 1e-10:
                    alpha[k] = np.clip(
                        np.sum(weights * pos_mask[:, k]) / denom_pos,
                        *self.parameter_clip,
                    )
                if denom_neg > 1e-10:
                    beta[k] = np.clip(
                        np.sum((1.0 - weights) * neg_mask[:, k]) / denom_neg,
                        *self.parameter_clip,
                    )

            if previous is not None and np.max(np.abs(weights - previous)) < 1e-5:
                break
            previous = weights

        self.source_sensitivities = alpha
        self.source_specificities = beta
        self.p_prior = p
        self.is_fitted = True
        return self

    def predict_proba(self, df_lfs):
        if not self.is_fitted:
            raise RuntimeError("Fit the label aggregator on train data before prediction")
        lf_matrix = df_lfs[self.lf_cols].to_numpy(dtype=int)

        if self.method == "consensus":
            probs = np.full(len(lf_matrix), 0.50, dtype=float)
            for i, row in enumerate(lf_matrix):
                non_abstain = row[row != 0]
                if len(non_abstain):
                    probs[i] = np.mean(non_abstain == 1)
            return probs

        return self._posterior(
            lf_matrix,
            self.p_prior,
            self.source_sensitivities,
            self.source_specificities,
        )

    def predict(self, df_lfs):
        probs = self.predict_proba(df_lfs)
        out = df_lfs.copy()
        out["y_prob"] = probs
        out["y_weak_consensus"] = (probs >= 0.50).astype(int)
        return out

    def fit_predict(self, df_lfs, lf_cols=DEFAULT_LF_COLS):
        return self.fit(df_lfs, lf_cols=lf_cols).predict(df_lfs)

    def parameters(self):
        if not self.is_fitted:
            raise RuntimeError("Label aggregator is not fitted")
        return {
            "method": self.method,
            "lf_cols": list(self.lf_cols),
            "p_prior": float(self.p_prior),
            "source_sensitivities": None
            if self.source_sensitivities is None
            else self.source_sensitivities.tolist(),
            "source_specificities": None
            if self.source_specificities is None
            else self.source_specificities.tolist(),
        }


def compute_lf_correlation_matrix(df_lfs, lf_cols=DEFAULT_LF_COLS):
    """Compute pairwise Spearman correlation between labeling functions."""
    matrix = df_lfs[lf_cols].values
    corr_df = pd.DataFrame(index=lf_cols, columns=lf_cols, dtype=float)

    for i, c1 in enumerate(lf_cols):
        for j, c2 in enumerate(lf_cols):
            if i == j:
                corr_df.loc[c1, c2] = 1.0
            else:
                rho, _ = spearmanr(matrix[:, i], matrix[:, j])
                corr_df.loc[c1, c2] = float(np.nan_to_num(rho))

    return corr_df
