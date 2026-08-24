"""Pipeline gán nhãn yếu: LF → Dawid–Skene / consensus → y_prob."""

from __future__ import annotations

import pandas as pd

from src.weak.lf import AspectLabelingFunctions
from src.weak.aggregator import (
    ProbabilisticLabelAggregator,
    compute_lf_correlation_matrix,
)

LF_COLS = ["lf_skill", "lf_sem", "lf_exp", "lf_role", "lf_loc"]


class WeakLabelPipeline:
    """Fit LF thresholds and the label model on train, then freeze both."""

    def __init__(self, pos_percentile: float = 75, neg_percentile: float = 25):
        self.lfs = AspectLabelingFunctions(
            pos_percentile=pos_percentile,
            neg_percentile=neg_percentile,
        )
        self.aggregators = {
            "dawid_skene": ProbabilisticLabelAggregator(method="dawid_skene"),
            "consensus": ProbabilisticLabelAggregator(method="consensus"),
        }
        self.is_fitted = False
        self.fitted_method = None

    def fit(self, df_train: pd.DataFrame) -> "WeakLabelPipeline":
        self.lfs.fit(df_train)
        self.is_fitted = True
        return self

    def transform_lfs(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("WeakLabelPipeline.fit(train) trước khi transform.")
        return self.lfs.transform(df)

    @staticmethod
    def _ensure_lf_columns(df_lfs: pd.DataFrame) -> pd.DataFrame:
        out = df_lfs.copy()
        if "lf_exp" not in out.columns and "exp_score" in out.columns:
            out["lf_exp"] = 0
            out.loc[out["exp_score"] >= 1.0, "lf_exp"] = 1
            out.loc[out["exp_score"] <= 0.5, "lf_exp"] = -1
        for col in LF_COLS:
            if col not in out.columns:
                out[col] = 0
        return out

    def fit_transform_train(
        self, df_train: pd.DataFrame, method: str = "dawid_skene"
    ) -> pd.DataFrame:
        if method not in self.aggregators:
            raise ValueError(f"Unknown aggregation method: {method}")
        self.fit(df_train)
        train_lfs = self._ensure_lf_columns(self.transform_lfs(df_train))
        self.aggregators[method].fit(train_lfs, lf_cols=LF_COLS)
        self.fitted_method = method
        return self.aggregators[method].predict(train_lfs)

    def aggregate(
        self, df_lfs: pd.DataFrame, method: str = "dawid_skene"
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Fit WeakLabelPipeline on train before aggregation")
        if self.fitted_method != method:
            raise RuntimeError(
                f"Aggregator '{method}' was not fitted on train; fitted method is "
                f"'{self.fitted_method}'"
            )
        out = self._ensure_lf_columns(df_lfs)
        return self.aggregators[method].predict(out)

    def transform(
        self, df: pd.DataFrame, method: str = "dawid_skene"
    ) -> pd.DataFrame:
        return self.aggregate(self.transform_lfs(df), method=method)

    def aggregator_parameters(self) -> dict:
        if self.fitted_method is None:
            raise RuntimeError("WeakLabelPipeline label model is not fitted")
        return self.aggregators[self.fitted_method].parameters()

    @property
    def aggregator_ds(self):
        return self.aggregators["dawid_skene"]

    @property
    def aggregator_consensus(self):
        return self.aggregators["consensus"]

    def lf_correlation(self, df_lfs: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in LF_COLS if c in df_lfs.columns]
        return compute_lf_correlation_matrix(df_lfs, lf_cols=cols)
