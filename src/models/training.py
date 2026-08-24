from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from src.data.loader import FEATURE_COLS
from src.models.pairing import pair_table_to_arrays
from src.models.skill_gap import SkillGapHead, build_gap_targets, skill_gap_loss


def set_deterministic_seed(seed: int) -> np.random.RandomState:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return np.random.RandomState(seed)


class FixedHeuristic:
    name = "H"

    def fit(self, _: pd.DataFrame) -> "FixedHeuristic":
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (
            0.30 * frame["loc_match"]
            + 0.25 * frame["skill_iou"]
            + 0.20 * frame["exp_score"]
            + 0.15 * frame["role_match"]
            + 0.10 * frame["desc_sem_sim"]
        ).to_numpy(float)


class PointwiseLogistic:
    name = "B1"

    def __init__(self, c: float = 1.0, seed: int = 42):
        self.model = LogisticRegression(
            C=c, max_iter=1000, random_state=seed, solver="lbfgs"
        )

    def fit(self, frame: pd.DataFrame) -> "PointwiseLogistic":
        self.model.fit(
            frame[FEATURE_COLS].to_numpy(float),
            frame["heuristic_label"].to_numpy(int),
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(frame[FEATURE_COLS].to_numpy(float))[:, 1]

    def parameters(self) -> dict:
        return {
            "intercept": float(self.model.intercept_[0]),
            "coefficients": {
                feature: float(value)
                for feature, value in zip(FEATURE_COLS, self.model.coef_[0])
            },
        }


class LinearSoftBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(len(FEATURE_COLS), 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs).squeeze(-1)


@dataclass
class PointwiseTrainingResult:
    model: LinearSoftBCE
    best_epoch: int
    best_validation_loss: float
    learning_rate: float
    batch_size: int
    stopped_early: bool

    def metadata(self) -> dict:
        return {
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "stopped_early": self.stopped_early,
        }


def _batches(size: int, batch_size: int, rng: np.random.RandomState):
    indices = np.arange(size)
    rng.shuffle(indices)
    for start in range(0, size, batch_size):
        yield indices[start : start + batch_size]


def train_linear_soft_bce(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    min_delta: float = 1e-6,
) -> PointwiseTrainingResult:
    rng = set_deterministic_seed(seed)
    model = LinearSoftBCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_x = torch.tensor(train[FEATURE_COLS].to_numpy(float), dtype=torch.float32)
    train_y = torch.tensor(train["y_prob"].to_numpy(float), dtype=torch.float32)
    validation_x = torch.tensor(
        validation[FEATURE_COLS].to_numpy(float), dtype=torch.float32
    )
    validation_y = torch.tensor(
        validation["y_prob"].to_numpy(float), dtype=torch.float32
    )
    if len(validation_x) == 0:
        raise ValueError("B2 requires an independent non-empty validation split")

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    final_epoch = 0
    for epoch in range(1, max_epochs + 1):
        final_epoch = epoch
        model.train()
        for batch in _batches(len(train_x), batch_size, rng):
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                model(train_x[batch]), train_y[batch]
            )
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                F.binary_cross_entropy_with_logits(
                    model(validation_x), validation_y
                ).item()
            )
        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("B2 produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    return PointwiseTrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        learning_rate=learning_rate,
        batch_size=batch_size,
        stopped_early=final_epoch < max_epochs,
    )


def predict_linear_soft_bce(
    model: LinearSoftBCE, frame: pd.DataFrame
) -> np.ndarray:
    model.eval()
    inputs = torch.tensor(frame[FEATURE_COLS].to_numpy(float), dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(model(inputs)).cpu().numpy()


class RankNetMLP(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(len(FEATURE_COLS), 32)
        self.fc2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.dropout = nn.Dropout(dropout)

    def hidden(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(F.relu(self.fc1(inputs)))
        return F.relu(self.fc2(hidden))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(inputs)).squeeze(-1)


@dataclass
class RankNetTrainingResult:
    model: RankNetMLP
    gap_head: SkillGapHead | None
    gap_vocabulary: list[str]
    best_epoch: int
    best_validation_rank_loss: float
    selected_checkpoint_gap_loss: float
    last_epoch_gap_loss: float
    learning_rate: float
    batch_size: int
    lambda_gap: float
    n_train_pairs: int
    n_validation_pairs: int
    stopped_early: bool

    def metadata(self) -> dict:
        return {
            "gap_vocabulary": self.gap_vocabulary,
            "best_epoch": self.best_epoch,
            "best_validation_rank_loss": self.best_validation_rank_loss,
            "selected_checkpoint_gap_loss": self.selected_checkpoint_gap_loss,
            "last_epoch_gap_loss": self.last_epoch_gap_loss,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "lambda_gap": self.lambda_gap,
            "n_train_pairs": self.n_train_pairs,
            "n_validation_pairs": self.n_validation_pairs,
            "stopped_early": self.stopped_early,
        }


def _rank_loss(
    model: RankNetMLP, preferred: torch.Tensor, nonpreferred: torch.Tensor
) -> torch.Tensor:
    return F.softplus(-(model(preferred) - model(nonpreferred))).mean()


def train_ranknet(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_pair_table: pd.DataFrame,
    validation_pair_table: pd.DataFrame,
    seed: int,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lambda_gap: float = 0.0,
    dropout: float = 0.1,
    min_delta: float = 1e-5,
) -> RankNetTrainingResult:
    if train_pair_table.empty or validation_pair_table.empty:
        raise ValueError("RankNet requires non-empty train and validation pair tables")
    rng = set_deterministic_seed(seed)
    model = RankNetMLP(dropout=dropout)
    train_preferred, train_nonpreferred = pair_table_to_arrays(
        train, train_pair_table
    )
    validation_preferred, validation_nonpreferred = pair_table_to_arrays(
        validation, validation_pair_table
    )
    train_preferred_t = torch.tensor(train_preferred, dtype=torch.float32)
    train_nonpreferred_t = torch.tensor(train_nonpreferred, dtype=torch.float32)
    validation_preferred_t = torch.tensor(
        validation_preferred, dtype=torch.float32
    )
    validation_nonpreferred_t = torch.tensor(
        validation_nonpreferred, dtype=torch.float32
    )

    gap_head = None
    gap_vocabulary: list[str] = []
    gap_targets_t = None
    all_train_x = None
    if lambda_gap > 0:
        if "job_skills" not in train or "user_skills" not in train:
            raise ValueError("M2 requires raw train skill columns; proxy fallback is forbidden")
        gap_targets, gap_vocabulary = build_gap_targets(train)
        if not gap_vocabulary:
            raise ValueError("M2 train skill-gap vocabulary is empty")
        gap_head = SkillGapHead(16, len(gap_vocabulary))
        gap_targets_t = torch.tensor(gap_targets, dtype=torch.float32)
        all_train_x = torch.tensor(
            train[FEATURE_COLS].to_numpy(float), dtype=torch.float32
        )

    parameters = list(model.parameters())
    if gap_head is not None:
        parameters += list(gap_head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    best_model_state = None
    best_gap_state = None
    best_validation_loss = float("inf")
    best_epoch = 0
    best_gap_loss = float("nan")
    last_gap_loss = float("nan")
    wait = 0
    final_epoch = 0

    for epoch in range(1, max_epochs + 1):
        final_epoch = epoch
        model.train()
        if gap_head is not None:
            gap_head.train()
        for batch in _batches(len(train_preferred_t), batch_size, rng):
            optimizer.zero_grad()
            loss = _rank_loss(
                model, train_preferred_t[batch], train_nonpreferred_t[batch]
            )
            if gap_head is not None:
                gap_loss = skill_gap_loss(
                    gap_head(model.hidden(all_train_x)), gap_targets_t
                )
                loss = loss + lambda_gap * gap_loss
            loss.backward()
            optimizer.step()

        model.eval()
        if gap_head is not None:
            gap_head.eval()
        with torch.no_grad():
            validation_loss = float(
                _rank_loss(
                    model, validation_preferred_t, validation_nonpreferred_t
                ).item()
            )
            if gap_head is not None:
                last_gap_loss = float(
                    skill_gap_loss(
                        gap_head(model.hidden(all_train_x)), gap_targets_t
                    ).item()
                )
        if validation_loss < best_validation_loss - min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_gap_state = (
                copy.deepcopy(gap_head.state_dict()) if gap_head is not None else None
            )
            best_gap_loss = last_gap_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_model_state is None:
        raise RuntimeError("RankNet produced no finite validation checkpoint")
    model.load_state_dict(best_model_state)
    if gap_head is not None:
        gap_head.load_state_dict(best_gap_state)
    return RankNetTrainingResult(
        model=model,
        gap_head=gap_head,
        gap_vocabulary=gap_vocabulary,
        best_epoch=best_epoch,
        best_validation_rank_loss=best_validation_loss,
        selected_checkpoint_gap_loss=best_gap_loss,
        last_epoch_gap_loss=last_gap_loss,
        learning_rate=learning_rate,
        batch_size=batch_size,
        lambda_gap=lambda_gap,
        n_train_pairs=len(train_pair_table),
        n_validation_pairs=len(validation_pair_table),
        stopped_early=final_epoch < max_epochs,
    )


def predict_ranknet(model: RankNetMLP, frame: pd.DataFrame) -> np.ndarray:
    model.eval()
    inputs = torch.tensor(frame[FEATURE_COLS].to_numpy(float), dtype=torch.float32)
    with torch.no_grad():
        return model(inputs).cpu().numpy()
