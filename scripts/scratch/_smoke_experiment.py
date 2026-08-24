# -*- coding: utf-8 -*-
"""Smoke-run core path of experiment_full.ipynb with 1 seed / tiny grid."""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression, Ridge

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.data import load_dataset, FEATURE_COLS
from src.weak import WeakLabelPipeline
from src.weak.pipeline import LF_COLS
from src.models.pairing import build_rank_pairs
from src.models.skill_gap import SkillGapHead, build_gap_targets, skill_gap_loss
from src.eval.metrics import ndcg_at_k, map_at_k, paired_bootstrap_test
from src.eval.perturbation import qualify_sensitivity


@dataclass
class ExpConfig:
    seeds: tuple = (42,)
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    feature_cols: tuple = tuple(FEATURE_COLS)
    lf_pos_percentile: float = 75.0
    lf_neg_percentile: float = 25.0
    aggregator: str = "dawid_skene"
    pair_margin: float = 0.05
    max_pairs_per_job: int = 100
    hidden1: int = 32
    hidden2: int = 16
    dropout: float = 0.1
    max_epochs: int = 15
    patience: int = 5
    lr_grid: tuple = (1e-3,)
    batch_grid: tuple = (32,)
    default_lr: float = 1e-3
    default_batch: int = 32
    lambda_gap_grid: tuple = (0.2,)
    default_lambda_gap: float = 0.2
    k_list: tuple = (5, 10)
    n_bootstrap: int = 100
    gold_path: str = "data/gold/human_validated_benchmark.csv"
    gold_rel_col: str = "human_relevance"
    out_dir: str = "reports/phase4a"


CFG = ExpConfig()
OUT_DIR = ROOT / CFG.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> np.random.RandomState:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.RandomState(seed)


df_raw = load_dataset(data_dir=str(ROOT / "data"), random_seed=42)
CAND_COL = "cand_id" if "cand_id" in df_raw.columns else "user_id"
if "heuristic_score" not in df_raw.columns:
    df_raw["heuristic_score"] = (
        0.30 * df_raw["loc_match"] + 0.25 * df_raw["skill_iou"] + 0.20 * df_raw["exp_score"]
        + 0.15 * df_raw["role_match"] + 0.10 * df_raw["desc_sem_sim"]
    )
if "heuristic_label" not in df_raw.columns:
    df_raw["heuristic_label"] = (df_raw["heuristic_score"] >= 0.45).astype(int)

gold = pd.read_csv(ROOT / CFG.gold_path)
if CAND_COL not in gold.columns:
    alt = "cand_id" if CAND_COL == "user_id" else "user_id"
    gold = gold.rename(columns={alt: CAND_COL})
GOLD_JOBS = sorted(gold["job_id"].unique().tolist())
feat_keys = ["job_id", CAND_COL] + list(FEATURE_COLS)
extra = [c for c in ["heuristic_score", "heuristic_label", "job_skills", "user_skills", "skill_iou"] if c in df_raw.columns]
gold_feat = gold.merge(
    df_raw[feat_keys + [c for c in extra if c not in feat_keys]].drop_duplicates(["job_id", CAND_COL]),
    on=["job_id", CAND_COL],
    how="inner",
)
print("gold_feat", len(gold_feat), "grades", sorted(gold_feat[CFG.gold_rel_col].unique()))


def job_disjoint_split(df, seed, holdout_jobs, train_ratio=0.80, val_ratio=0.10):
    rng = np.random.RandomState(seed)
    jobs = np.array(sorted(set(df["job_id"].unique()) - set(holdout_jobs)))
    rng.shuffle(jobs)
    n = len(jobs)
    n_train = max(1, int(train_ratio * n))
    n_val = max(1, int(val_ratio * n))
    train_jobs = jobs[:n_train]
    val_jobs = jobs[n_train:n_train + n_val]
    test_jobs = jobs[n_train + n_val:]
    if len(test_jobs) == 0:
        test_jobs = train_jobs[-1:]
        train_jobs = train_jobs[:-1]
    return (
        df[df["job_id"].isin(train_jobs)].copy(),
        df[df["job_id"].isin(val_jobs)].copy(),
        df[df["job_id"].isin(test_jobs)].copy(),
    )


class ModelH:
    name = "H"
    def fit(self, df_train):
        return self
    def predict(self, df):
        return (
            0.30 * df["loc_match"].values + 0.25 * df["skill_iou"].values
            + 0.20 * df["exp_score"].values + 0.15 * df["role_match"].values
            + 0.10 * df["desc_sem_sim"].values
        )


class ModelB1:
    name = "B1"
    def __init__(self):
        self.clf = LogisticRegression(C=1.0, max_iter=500, random_state=0)
    def fit(self, df_train):
        self.clf.fit(df_train[list(CFG.feature_cols)].values, df_train["heuristic_label"].values.astype(int))
        return self
    def predict(self, df):
        return self.clf.predict_proba(df[list(CFG.feature_cols)].values)[:, 1]


class ModelB2:
    name = "B2"
    def __init__(self):
        self.reg = Ridge(alpha=1.0)
    def fit(self, df_train):
        self.reg.fit(df_train[list(CFG.feature_cols)].values, df_train["y_prob"].values.astype(float))
        return self
    def predict(self, df):
        return self.reg.predict(df[list(CFG.feature_cols)].values)


class RankNetMLP(nn.Module):
    def __init__(self, input_dim=5, h1=32, h2=16, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        return self.out(h).squeeze(-1)
    def hidden(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return F.relu(self.fc2(h))


def eval_ndcg_map(df, scores, rel_col, k_list=(5, 10)):
    tmp = df.copy()
    tmp["_s"] = scores
    out = {}
    for k in k_list:
        nds, mps = [], []
        for _, g in tmp.groupby("job_id"):
            yt, ys = g[rel_col].values, g["_s"].values
            nds.append(ndcg_at_k(yt, ys, k=k))
            mps.append(map_at_k(yt, ys, k=k))
        out[f"ndcg@{k}"] = float(np.mean(nds)) if nds else float("nan")
        out[f"map@{k}"] = float(np.mean(mps)) if mps else float("nan")
    return out


def predict_net(net, df):
    net.eval()
    X = torch.tensor(df[list(CFG.feature_cols)].values, dtype=torch.float32)
    with torch.no_grad():
        return net(X).cpu().numpy()


def _batches(n, batch_size, rng):
    idx = np.arange(n)
    rng.shuffle(idx)
    for s in range(0, n, batch_size):
        yield idx[s:s + batch_size]


def train_m1(df_train, df_val, seed, lr, batch_size):
    rng = set_seed(seed)
    Xi, Xj = build_rank_pairs(
        df_train, score_col="y_prob", max_pairs_per_job=CFG.max_pairs_per_job,
        margin=CFG.pair_margin, rng=np.random.RandomState(seed),
    )
    net = RankNetMLP(len(CFG.feature_cols), CFG.hidden1, CFG.hidden2, CFG.dropout)
    if len(Xi) == 0:
        return net, 0.0, 0
    ti = torch.tensor(Xi, dtype=torch.float32)
    tj = torch.tensor(Xj, dtype=torch.float32)
    opt = optim.Adam(net.parameters(), lr=lr)
    best_state, best_ndcg, best_ep, wait = None, -1.0, 0, 0
    for epoch in range(1, CFG.max_epochs + 1):
        net.train()
        for bidx in _batches(len(ti), batch_size, rng):
            opt.zero_grad()
            loss = F.softplus(-(net(ti[bidx]) - net(tj[bidx]))).mean()
            loss.backward()
            opt.step()
        cur = eval_ndcg_map(df_val, predict_net(net, df_val), "y_prob", (5,))["ndcg@5"]
        if cur > best_ndcg + 1e-5:
            best_ndcg, best_ep, wait = cur, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= CFG.patience:
                break
    if best_state:
        net.load_state_dict(best_state)
    return net, best_ndcg, best_ep


def train_m2(df_train, df_val, seed, lr, batch_size, lambda_gap):
    rng = set_seed(seed)
    Xi, Xj = build_rank_pairs(
        df_train, score_col="y_prob", max_pairs_per_job=CFG.max_pairs_per_job,
        margin=CFG.pair_margin, rng=np.random.RandomState(seed),
    )
    Y_gap, vocab = build_gap_targets(df_train)
    net = RankNetMLP(len(CFG.feature_cols), CFG.hidden1, CFG.hidden2, CFG.dropout)
    gap_head = SkillGapHead(CFG.hidden2, Y_gap.shape[1])
    if len(Xi) == 0:
        return net, 0.0, 0
    ti = torch.tensor(Xi, dtype=torch.float32)
    tj = torch.tensor(Xj, dtype=torch.float32)
    X_all = torch.tensor(df_train[list(CFG.feature_cols)].values, dtype=torch.float32)
    Y_all = torch.tensor(Y_gap, dtype=torch.float32)
    opt = optim.Adam(list(net.parameters()) + list(gap_head.parameters()), lr=lr)
    best_state, best_gap, best_ndcg, best_ep, wait = None, None, -1.0, 0, 0
    for epoch in range(1, CFG.max_epochs + 1):
        net.train(); gap_head.train()
        for bidx in _batches(len(ti), batch_size, rng):
            opt.zero_grad()
            l_rank = F.softplus(-(net(ti[bidx]) - net(tj[bidx]))).mean()
            l_gap = skill_gap_loss(gap_head(net.hidden(X_all)), Y_all)
            (l_rank + lambda_gap * l_gap).backward()
            opt.step()
        cur = eval_ndcg_map(df_val, predict_net(net, df_val), "y_prob", (5,))["ndcg@5"]
        if cur > best_ndcg + 1e-5:
            best_ndcg, best_ep, wait = cur, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            best_gap = {k: v.detach().cpu().clone() for k, v in gap_head.state_dict().items()}
        else:
            wait += 1
            if wait >= CFG.patience:
                break
    if best_state:
        net.load_state_dict(best_state)
        gap_head.load_state_dict(best_gap)
    return net, best_ndcg, best_ep


class Predictor:
    def __init__(self, fn):
        self._fn = fn
    def predict(self, df):
        return self._fn(df)


seed = CFG.seeds[0]
set_seed(seed)
df_train, df_val, df_test = job_disjoint_split(df_raw, seed, GOLD_JOBS)
pipe = WeakLabelPipeline(CFG.lf_pos_percentile, CFG.lf_neg_percentile)
df_train_w = pipe.fit_transform_train(df_train, method=CFG.aggregator)
df_val_w = pipe.aggregate(pipe.transform_lfs(df_val), method=CFG.aggregator)
df_gold_w = pipe.aggregate(pipe.transform_lfs(gold_feat), method=CFG.aggregator).copy()
df_gold_w[CFG.gold_rel_col] = gold_feat[CFG.gold_rel_col].values
print("y_prob mean", df_train_w["y_prob"].mean(), "LF", LF_COLS)

rows = []
models = {}
for cls, name in [(ModelH, "H"), (ModelB1, "B1"), (ModelB2, "B2")]:
    t0 = time.time()
    m = cls().fit(df_train_w)
    met = eval_ndcg_map(df_gold_w, m.predict(df_gold_w), CFG.gold_rel_col, CFG.k_list)
    rows.append({"model": name, "seed": seed, **met, "training_time": time.time() - t0})
    models[name] = Predictor(m.predict)
    print(name, met)

net1, _, ep1 = train_m1(df_train_w, df_val_w, seed, CFG.default_lr, CFG.default_batch)
met = eval_ndcg_map(df_gold_w, predict_net(net1, df_gold_w), CFG.gold_rel_col, CFG.k_list)
rows.append({"model": "M1", "seed": seed, **met, "training_time": 0.0, "best_epoch": ep1})
models["M1"] = Predictor(lambda d, n=net1: predict_net(n, d))
print("M1", met, "ep", ep1)

net2, _, ep2 = train_m2(df_train_w, df_val_w, seed, CFG.default_lr, CFG.default_batch, CFG.default_lambda_gap)
met = eval_ndcg_map(df_gold_w, predict_net(net2, df_gold_w), CFG.gold_rel_col, CFG.k_list)
rows.append({"model": "M2", "seed": seed, **met, "training_time": 0.0, "best_epoch": ep2})
models["M2"] = Predictor(lambda d, n=net2: predict_net(n, d))
print("M2", met, "ep", ep2)

qs = {
    "M1": qualify_sensitivity(df_gold_w, models["M1"].predict),
    "M2": qualify_sensitivity(df_gold_w, models["M2"].predict),
}
print("QualSens", json.dumps(qs, indent=2))

ci = paired_bootstrap_test(
    df_gold_w, models["H"], models["M1"], metric_fn=ndcg_at_k, k=5,
    target_col=CFG.gold_rel_col, n_bootstraps=CFG.n_bootstrap, seed=seed,
)
print("M1-H bootstrap", {k: ci[k] for k in ["mean_delta", "ci_95_low", "ci_95_high", "p_value"]})

results_df = pd.DataFrame(rows)
results_df.to_csv(OUT_DIR / "results_all_runs.csv", index=False)
pd.DataFrame([{
    "comparison": "M1_vs_H",
    "mean_delta": ci["mean_delta"],
    "ci_95_low": ci["ci_95_low"],
    "ci_95_high": ci["ci_95_high"],
    "p_value": ci["p_value"],
}]).to_csv(OUT_DIR / "confidence_interval.csv", index=False)
print("SMOKE OK")
print(results_df.to_string(index=False))
