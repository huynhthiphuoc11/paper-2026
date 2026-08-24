"""Skill-gap auxiliary — Gap = Skill_JD \\ Skill_CV (set-difference)."""

from __future__ import annotations

from typing import Iterable, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


_SKILL_JUNK = {
    ".",
    "..",
    "...",
    "....",
    "-",
    "--",
    "---",
    "_",
    "__",
    "•",
    "·",
    "*",
    "/",
    "\\",
    "|",
    ":",
    ";",
    ",",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
}


def _is_valid_skill_token(tok: str) -> bool:
    t = tok.strip().lower()
    if not t or t in _SKILL_JUNK:
        return False
    # drop punctuation-only / ellipsis-like tokens
    if all(ch in ".-_•·*/\\|:;," for ch in t):
        return False
    if len(t) < 2:
        return False
    return True


def parse_skill_set(raw) -> Set[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return set()
    if isinstance(raw, (set, list, tuple)):
        return {str(s).strip().lower() for s in raw if _is_valid_skill_token(str(s))}
    text = str(raw)
    for sep in [";", "|", ",", "/"]:
        text = text.replace(sep, " ")
    return {t.strip().lower() for t in text.split() if _is_valid_skill_token(t)}


def skill_gap_vector(
    skills_jd: Iterable[str],
    skills_cv: Iterable[str],
    vocab: list[str],
) -> np.ndarray:
    """Multi-hot: 1 nếu skill thuộc JD nhưng không thuộc CV."""
    jd, cv = set(skills_jd), set(skills_cv)
    missing = jd - cv
    return np.asarray([1.0 if s in missing else 0.0 for s in vocab], dtype=np.float32)


def build_gap_targets(
    df: pd.DataFrame,
    jd_skill_col: str = "job_skills",
    cv_skill_col: str = "user_skills",
    vocab: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Trả (Y_gap [n, |V|], vocab).
    Nếu thiếu cột skill thô, fallback: dùng (1 - skill_iou) scalar → Y shape (n, 1).
    """
    if jd_skill_col not in df.columns or cv_skill_col not in df.columns:
        y = (1.0 - df["skill_iou"].fillna(0).values.astype(np.float32)).reshape(-1, 1)
        return y, ["missing_overlap_proxy"]

    jd_sets = [parse_skill_set(v) for v in df[jd_skill_col]]
    cv_sets = [parse_skill_set(v) for v in df[cv_skill_col]]
    if vocab is None:
        vocab_set: Set[str] = set()
        for s in jd_sets:
            vocab_set |= s
        vocab = sorted(vocab_set)[:256]  # trần vocabulary — tránh phình
    Y = np.stack(
        [skill_gap_vector(j, c, vocab) for j, c in zip(jd_sets, cv_sets)],
        axis=0,
    )
    return Y, vocab


class SkillGapHead(nn.Module):
    """Đầu phụ: từ hidden/score-input dự đoán multi-hot gap."""

    def __init__(self, input_dim: int, gap_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, gap_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


def skill_gap_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)
