"""Controlled qualification perturbation — protocol RQ2 (DE_CUONG_CHOT §11.2)."""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import pandas as pd

PerturbKind = Literal["skill", "exp", "domain"]


def apply_isolated_perturbation(
    row: pd.Series,
    kind: PerturbKind,
    skill_drop: float = 0.35,
    exp_drop: float = 0.4,
    domain_role_cap: float = 0.15,
) -> pd.Series:
    """
    Tạo biến thể isolated từ một cặp CV–JD (chỉ đổi một cơ chế).
    Thao tác trên feature bảng đã có — không sinh lại văn bản.
    """
    out = row.copy()
    if kind == "skill":
        out["skill_iou"] = max(0.0, float(row["skill_iou"]) - skill_drop)
    elif kind == "exp":
        out["exp_score"] = max(0.0, float(row["exp_score"]) - exp_drop)
    elif kind == "domain":
        out["role_match"] = min(float(row["role_match"]), domain_role_cap)
        out["desc_sem_sim"] = min(float(row.get("desc_sem_sim", 0.0)), domain_role_cap)
    else:
        raise ValueError(f"Unknown perturbation kind: {kind}")
    return out


def expected_order_held(score_v0: float, score_vx: float) -> bool:
    return float(score_v0) > float(score_vx)


def qualify_sensitivity(
    df_v0: pd.DataFrame,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    kinds: tuple[PerturbKind, ...] = ("skill", "exp", "domain"),
) -> dict:
    """
    QualSens_x = (1/N) Σ 1[score(V0) > score(V_x)], tách theo nhóm.
    """
    scores_v0 = predict_fn(df_v0)
    report = {}
    for kind in kinds:
        held = []
        rows = [apply_isolated_perturbation(df_v0.iloc[i], kind) for i in range(len(df_v0))]
        df_vx = pd.DataFrame(rows)
        scores_vx = predict_fn(df_vx)
        for s0, sx in zip(scores_v0, scores_vx):
            held.append(1.0 if expected_order_held(s0, sx) else 0.0)
        report[kind] = {
            "QualSens": float(np.mean(held)) if held else float("nan"),
            "n": len(held),
        }
    return report
