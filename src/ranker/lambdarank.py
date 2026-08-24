"""LightGBM LambdaRank baseline with an explicit dependency error, never silent fallback."""
from __future__ import annotations
import numpy as np
import pandas as pd

class LambdaRankRanker:
    name = "LambdaRank"
    def __init__(self, feature_cols, seed=42, n_estimators=200, learning_rate=.05, num_leaves=15):
        self.feature_cols, self.seed = list(feature_cols), seed
        self.kwargs = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                           random_state=seed, objective="lambdarank", metric="ndcg")
        self.model = None

    def fit(self, df: pd.DataFrame, label_col: str):
        try:
            from lightgbm import LGBMRanker
        except ImportError as exc:
            raise ImportError("LambdaRank benchmark requires optional dependency lightgbm. Install lightgbm>=4.0.") from exc
        ordered = df.sort_values("job_id", kind="stable")
        groups = ordered.groupby("job_id", sort=False).size().tolist()
        if min(groups, default=0) < 2:
            raise ValueError("LambdaRank requires at least two candidates per job.")
        self.model = LGBMRanker(**self.kwargs)
        self.model.fit(ordered[self.feature_cols], ordered[label_col], group=groups)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None: raise RuntimeError("Call fit before predict.")
        return self.model.predict(df[self.feature_cols])
