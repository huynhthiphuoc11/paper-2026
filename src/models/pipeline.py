"""
Chuỗi mô hình theo DE_CUONG_CHOT:

H  → trọng số tay
B1 → pointwise BCE (learned weighting)
B2 → pointwise soft y_prob
M1 → RankNet
M2 → RankNet + skill-gap (khung; train đầy đủ ở phase 4b)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from src.data.loader import FEATURE_COLS
from src.models.legacy import ModelH_Heuristic, RankScoringNet
from src.models.pairing import build_rank_pairs
from src.models.skill_gap import SkillGapHead, build_gap_targets, skill_gap_loss


class H_FixedHeuristic(ModelH_Heuristic):
    """Baseline Huynh — không học."""
    name = "H"


class B1_PointwiseBCE:
    """Learned weighting tuyến tính (Logistic trên heuristic_label hoặc nhãn cứng)."""

    name = "B1"

    def __init__(self, label_col: str = "heuristic_label"):
        self.label_col = label_col
        self.clf = LogisticRegression(C=1.0, max_iter=500)

    def fit(self, df_train: pd.DataFrame):
        X = df_train[FEATURE_COLS].values
        y = df_train[self.label_col].values
        self.clf.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(df[FEATURE_COLS].values)[:, 1]

    @property
    def weights_(self) -> dict:
        coef = self.clf.coef_.ravel()
        return {f: float(w) for f, w in zip(FEATURE_COLS, coef)}


class B2_PointwiseSoft:
    """Pointwise soft BCEWithLogits trên y_prob (Dawid–Skene) — RESEARCH_AUDIT C4."""

    name = "B2"

    def __init__(self, target_col: str = "y_prob", lr: float = 1e-2, epochs: int = 200, seed: int = 0):
        self.target_col = target_col
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.lin: nn.Linear | None = None

    def fit(self, df_train: pd.DataFrame):
        torch.manual_seed(self.seed)
        X = torch.tensor(df_train[FEATURE_COLS].values, dtype=torch.float32)
        y = torch.tensor(df_train[self.target_col].values, dtype=torch.float32)
        self.lin = nn.Linear(X.shape[1], 1)
        opt = optim.Adam(self.lin.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(self.lin(X).squeeze(-1), y)
            loss.backward()
            opt.step()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.lin is not None
        self.lin.eval()
        X = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(self.lin(X).squeeze(-1)).cpu().numpy()


class M1_RankNet:
    """RankNet pairwise trên 5 feature; pairs từ y_prob."""

    name = "M1"

    def __init__(
        self,
        epochs: int = 40,
        lr: float = 1e-3,
        hidden_dim: int = 16,
        max_pairs_per_job: int = 100,
        pair_margin: float = 0.05,
        score_col: str = "y_prob",
    ):
        self.epochs = epochs
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.max_pairs_per_job = max_pairs_per_job
        self.pair_margin = pair_margin
        self.score_col = score_col
        self.net = RankScoringNet(input_dim=len(FEATURE_COLS), hidden_dim=hidden_dim)

    def fit(self, df_train: pd.DataFrame):
        X_i, X_j = build_rank_pairs(
            df_train,
            score_col=self.score_col,
            max_pairs_per_job=self.max_pairs_per_job,
            margin=self.pair_margin,
        )
        if len(X_i) == 0:
            return self
        ti = torch.tensor(X_i, dtype=torch.float32)
        tj = torch.tensor(X_j, dtype=torch.float32)
        opt = optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = F.softplus(-(self.net(ti) - self.net(tj))).mean()
            loss.backward()
            opt.step()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        X = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
        with torch.no_grad():
            return self.net(X).cpu().numpy()


class M2_RankNetSkillGap(M1_RankNet):
    """
    RankNet + auxiliary skill-gap.
    L = L_rank + λ * L_gap
    """

    name = "M2"

    def __init__(self, lambda_gap: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.lambda_gap = lambda_gap
        self.gap_head: SkillGapHead | None = None
        self.gap_vocab: list[str] = []

    def fit(self, df_train: pd.DataFrame):
        X_i, X_j = build_rank_pairs(
            df_train,
            score_col=self.score_col,
            max_pairs_per_job=self.max_pairs_per_job,
            margin=self.pair_margin,
        )
        Y_gap, vocab = build_gap_targets(df_train)
        self.gap_vocab = vocab
        self.gap_head = SkillGapHead(
            input_dim=self.hidden_dim,
            gap_dim=Y_gap.shape[1],
        )
        # Dùng hidden qua net.net[0]
        params = list(self.net.parameters()) + list(self.gap_head.parameters())
        opt = optim.Adam(params, lr=self.lr)

        if len(X_i) == 0:
            return self

        ti = torch.tensor(X_i, dtype=torch.float32)
        tj = torch.tensor(X_j, dtype=torch.float32)
        # Gap targets căn theo hàng df_train — dùng toàn bộ pointwise gap
        X_all = torch.tensor(df_train[FEATURE_COLS].values, dtype=torch.float32)
        Y_all = torch.tensor(Y_gap, dtype=torch.float32)

        self.net.train()
        self.gap_head.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            s_i = self.net(ti)
            s_j = self.net(tj)
            l_rank = F.softplus(-(s_i - s_j)).mean()
            # Hidden = ReLU(Linear)
            h = self.net.net[0](X_all)
            h = self.net.net[1](h)
            gap_logits = self.gap_head(h)
            l_gap = skill_gap_loss(gap_logits, Y_all)
            loss = l_rank + self.lambda_gap * l_gap
            loss.backward()
            opt.step()
        return self


def build_default_models() -> dict:
    """Registry H / B1 / B2 / M1 / M2 theo đề cương."""
    return {
        "H": H_FixedHeuristic(),
        "B1": B1_PointwiseBCE(),
        "B2": B2_PointwiseSoft(),
        "M1": M1_RankNet(),
        "M2": M2_RankNetSkillGap(),
    }
