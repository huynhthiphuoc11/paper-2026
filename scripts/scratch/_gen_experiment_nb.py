# -*- coding: utf-8 -*-
"""Generate notebooks/experiment_full.ipynb after RESEARCH_AUDIT + external audit fixes."""
from __future__ import annotations

import json
from pathlib import Path

raise RuntimeError(
    "Legacy notebook generator is disabled. experiment_full.ipynb is an "
    "artifact-only report for results/ from scripts/run_paper_experiment.py."
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "experiment_full.ipynb"
CELL_INDEX = 0


def _next_cell_id() -> str:
    global CELL_INDEX
    cell_id = f"cell-{CELL_INDEX:02d}"
    CELL_INDEX += 1
    return cell_id


def md(source: str) -> dict:
    lines = source.strip("\n").split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {"id": _next_cell_id(), "cell_type": "markdown", "metadata": {}, "source": src}


def code(source: str) -> dict:
    lines = source.strip("\n").split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {
        "id": _next_cell_id(),
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


cells: list[dict] = []

cells.append(md(r"""
# Experimental Framework — Vietnamese CV–JD Weakly Supervised LTR

**Audit:** [`docs/RESEARCH_AUDIT.md`](../docs/RESEARCH_AUDIT.md) · **Đề cương:** [`DE_CUONG_CHOT.md`](../DE_CUONG_CHOT.md)

## 01. Research Hypothesis

**Claim được phép**

> Learning-based ranking functions can outperform manually designed weighted matching scores under weak supervision (no hiring outcomes).

**Không claim:** trọng số tối ưu toàn cục · dự đoán hired/rejected · production ranking · LF/gold = ground truth · RankNet MLP = “học trọng số”.

**Giả thuyết vận hành**

| Bước | Giả thuyết kiểm chứng | Model |
|---|---|---|
| 1 | Learned linear weights cải thiện gold ranking vs trọng số tay | H → B1 |
| 2 | Soft weak labels hữu ích hơn / khác hard heuristic labels | B1 → B2 |
| 3 | Pairwise RankNet cải thiện vs pointwise | B2 → M1 (RQ1) |
| 4 | Skill-gap phụ cải thiện QualSens và/hoặc NDCG | M1 → M2 (RQ2) |

Hai RQ **độc lập**. M2 ↑ NDCG nhưng QualSens không ↑ → *chỉ* kết luận cải thiện ranking metric, chưa chứng minh robustness năng lực.
"""))

cells.append(md(r"""
## 02. Experimental Assumptions (sau audit)

1. **Cùng feature space:** train subsample và gold dùng **cùng** adapter TF–IDF cache (`prepare_feature_space`). Không BGE/BERT/BM25.
2. **Weak labels phụ thuộc feature:** LF = ngưỡng trên cùng `x` → `y_prob` không độc lập với đầu vào. Đánh giá khóa = gold độc lập.
3. **B1 target lệch H:** `heuristic_label = 1[H≥0.45]`. B1 chỉ ablation learned weighting.
4. **Model selection không đụng gold:** early-stop / HP theo **val pair-loss** `L_rank`.
5. **B2 = soft BCEWithLogits** trên `y_prob` — không Ridge MSE.
6. **Gold:** *graded evaluation set* 0–3 (`relevance`); một annotator; không gọi GT; ~12 JD → power thấp.
7. **CI chính:** seed-level paired delta (5 seeds). Job-bootstrap trong 1 seed chỉ là diagnostic.
8. **Seed kép:** cùng `seed` vừa shuffle JD split vừa init trọng số / pair sample trong grid HP → within-seed HP fair; across-seed đo đồng thời split+init variance.
9. **Kết luận:** CI 95% chứa 0 → không kết luận cải thiện chắc chắn.
10. **Val pairs:** `build_rank_pairs_robust` hạ margin đến 0; nếu vẫn rỗng thì fail-fast — không tune/early-stop trên train.
"""))

cells.append(md("## Environment & Imports"))
cells.append(code(r"""
from __future__ import annotations

import os, sys, json, time, warnings, hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from IPython.display import display

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import src.data.loader as loader_module
import src.weak.aggregator as aggregator_module
import src.models.skill_gap as skill_gap_module
from src.data.loader import FEATURE_COLS, RealKaggleDatasetAdapter
from src.weak import WeakLabelPipeline
from src.weak.pipeline import LF_COLS
from src.models.pairing import build_rank_pairs, build_rank_pairs_robust
from src.models.skill_gap import SkillGapHead, build_gap_targets, parse_skill_set, skill_gap_loss
from src.eval.metrics import ndcg_at_k, map_at_k, paired_bootstrap_test
from src.eval.perturbation import qualify_sensitivity

EXPECTED_FEATURE_COLS = (
    "loc_match",
    "skill_iou",
    "exp_score",
    "role_match",
    "desc_sem_sim",
)
CRITICAL_MODULES = {
    "src.data.loader": loader_module,
    "src.weak.aggregator": aggregator_module,
    "src.models.skill_gap": skill_gap_module,
}

def source_fingerprint(module) -> dict:
    path = Path(module.__file__).resolve()
    return {
        "path": str(path),
        "sha256_12": hashlib.sha256(path.read_bytes()).hexdigest()[:12],
    }

SOURCE_FINGERPRINTS = {
    name: source_fingerprint(module)
    for name, module in CRITICAL_MODULES.items()
}
assert tuple(FEATURE_COLS) == EXPECTED_FEATURE_COLS, (
    "Stale or incompatible loader source: expected the locked five-feature contract, "
    f"got {FEATURE_COLS}"
)
assert parse_skill_set("python .. java ... .._• sql") == {"python", "java", "sql"}, (
    "Stale skill_gap source: punctuation-only tokens were not removed"
)
for api_name in ("fit", "predict_proba", "predict", "parameters"):
    assert hasattr(aggregator_module.ProbabilisticLabelAggregator, api_name), (
        f"Stale aggregator source: missing train-only API '{api_name}'"
    )

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="muted")
print("ROOT", ROOT)
print("FEATURE_COLS", FEATURE_COLS)
print("SOURCE_FINGERPRINTS", json.dumps(SOURCE_FINGERPRINTS, indent=2))
print("Device note: cudnn deterministic flags only apply when CUDA is available.")
"""))

cells.append(md("## Configuration (paper mode)"))
cells.append(code(r"""
@dataclass
class ExpConfig:
    seeds: tuple = (42, 52, 62, 72, 82)
    train_ratio: float = 0.80
    val_ratio: float = 0.15
    feature_cols: tuple = tuple(FEATURE_COLS)
    n_jobs_sample: int = 160
    n_cands_sample: int = 240
    candidates_per_job: int = 100
    df_threshold: float = 0.15
    lf_pos_percentile: float = 75.0
    lf_neg_percentile: float = 25.0
    aggregator: str = "dawid_skene"
    pair_margin: float = 0.02
    max_pairs_per_job: int = 150
    hidden1: int = 32
    hidden2: int = 16
    dropout: float = 0.1
    max_epochs: int = 100
    patience: int = 10
    lr_grid: tuple = (1e-2, 1e-3, 1e-4)
    batch_grid: tuple = (16, 32, 64)
    default_lr: float = 1e-3
    default_batch: int = 32
    lambda_gap_grid: tuple = (0.05, 0.1, 0.2, 0.4)
    default_lambda_gap: float = 0.2
    k_list: tuple = (5, 10)
    n_bootstrap: int = 1000
    gold_path: str = "data/gold/human_validated_benchmark_graded_0_3.csv"
    gold_rel_col: str = "relevance"
    out_dir: str = "reports/phase4a_corrected"
    fig_dir: str = "reports/phase4a_corrected/figures"
    smoke: bool = False  # MUST stay False to lock paper numbers

CFG = ExpConfig()
if os.environ.get("EXPERIMENT_SMOKE") == "1":
    CFG = ExpConfig(
        seeds=(42,), lr_grid=(1e-3,), batch_grid=(32,),
        lambda_gap_grid=(0.2,), max_epochs=20, n_bootstrap=200,
        out_dir="reports/smoke_corrected", fig_dir="reports/smoke_corrected/figures",
        smoke=True,
    )
if CFG.smoke:
    print("SMOKE MODE | authoritative pipeline | seeds", CFG.seeds)
else:
    print("PAPER MODE | seeds", CFG.seeds, "| gold", CFG.gold_path)

OUT_DIR = ROOT / CFG.out_dir
FIG_DIR = ROOT / CFG.fig_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
assert tuple(CFG.feature_cols) == EXPECTED_FEATURE_COLS
(OUT_DIR / "runtime_preflight.json").write_text(
    json.dumps(
        {
            "root": str(ROOT.resolve()),
            "feature_cols": list(CFG.feature_cols),
            "source_fingerprints": SOURCE_FINGERPRINTS,
            "skill_parser_probe": sorted(parse_skill_set("python .. java ... .._• sql")),
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

def set_seed(seed: int) -> np.random.RandomState:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return np.random.RandomState(seed)

print(json.dumps(asdict(CFG), indent=2, default=list))
"""))

cells.append(md(r"""
## 03. Dataset Audit

Kiểm tra trước train: quy mô, missing, trùng cặp. Subsample Kaggle seed=42 (`n_jobs_sample`×`n_cands_sample`, tối đa `candidates_per_job` CV/JD).
"""))
cells.append(code(r"""
ADAPTER = RealKaggleDatasetAdapter(
    data_dir=str(ROOT / "data"),
    random_seed=42,
    df_threshold=CFG.df_threshold,
)
df_raw = ADAPTER.load_and_preprocess(
    n_jobs=CFG.n_jobs_sample,
    n_candidates=CFG.n_cands_sample,
    candidates_per_job=CFG.candidates_per_job,
)
CAND_COL = "cand_id" if "cand_id" in df_raw.columns else "user_id"
assert all(c in df_raw.columns for c in FEATURE_COLS)

audit = {
    "n_pairs": int(len(df_raw)),
    "n_jobs": int(df_raw["job_id"].nunique()),
    "n_cands": int(df_raw[CAND_COL].nunique()),
    "pairs_per_job_mean": float(df_raw.groupby("job_id").size().mean()),
    "pairs_per_job_min": int(df_raw.groupby("job_id").size().min()),
    "pairs_per_job_max": int(df_raw.groupby("job_id").size().max()),
    "dup_pair_keys": int(df_raw.duplicated(["job_id", CAND_COL]).sum()),
    "nan_features": {c: int(df_raw[c].isna().sum()) for c in FEATURE_COLS},
    "heuristic_pos_rate": float(df_raw["heuristic_label"].mean()),
    "has_skill_text_cols": bool({"job_skills", "user_skills"} <= set(df_raw.columns)),
}
print(json.dumps(audit, indent=2, ensure_ascii=False))
display(df_raw[list(FEATURE_COLS)].describe().T[["min", "mean", "max"]])
display(df_raw[list(FEATURE_COLS)].corr().round(3))
assert audit["dup_pair_keys"] == 0
assert sum(audit["nan_features"].values()) == 0
"""))

cells.append(md(r"""
## 04. Leakage Check

Cấm: cùng JD train/val/test; gold JD trong train; fit LF trên gold/test; dùng gold để tune.

Cho phép: cùng ứng viên (`cand_id`) ở nhiều split (job-disjoint). **Không claim** generalization sang CV hoàn toàn mới — limitation §17.
"""))
cells.append(code(r"""
def job_disjoint_split(df, seed, holdout_jobs, train_ratio=0.80, val_ratio=0.10):
    rng = np.random.RandomState(seed)
    jobs = np.array(sorted(set(df["job_id"].unique()) - set(holdout_jobs)))
    rng.shuffle(jobs)
    n = len(jobs)
    n_train = max(1, int(train_ratio * n))
    n_val = max(1, int(val_ratio * n))
    train_jobs, val_jobs = jobs[:n_train], jobs[n_train:n_train + n_val]
    test_jobs = jobs[n_train + n_val:]
    if len(test_jobs) == 0:
        test_jobs = train_jobs[-1:]; train_jobs = train_jobs[:-1]
    s_tr, s_va, s_te = set(train_jobs), set(val_jobs), set(test_jobs)
    hold = set(holdout_jobs)
    assert s_tr.isdisjoint(s_va) and s_tr.isdisjoint(s_te) and s_va.isdisjoint(s_te)
    assert hold.isdisjoint(s_tr | s_va | s_te)
    return (
        df[df["job_id"].isin(train_jobs)].copy(),
        df[df["job_id"].isin(val_jobs)].copy(),
        df[df["job_id"].isin(test_jobs)].copy(),
        {"train_jobs": sorted(s_tr), "val_jobs": sorted(s_va), "test_jobs": sorted(s_te)},
    )

print("job_disjoint_split ready")
"""))

cells.append(md(r"""
## 05. Gold Evaluation Loader + Shared Feature Space

*Graded evaluation set* 0–3. Feature gold tính bằng **cùng** `ADAPTER` đã fit TF–IDF cho `df_raw` (không fit lại vectorizer riêng).

Assertion: graded IDs ánh xạ đúng raw JD/CV đã annotate, coverage 100/100, và graded JD không xuất hiện trong sampled training pool. Graded features dùng chính TF–IDF transformers đã fit trên paper subsample.
"""))
cells.append(code(r"""
gold = pd.read_csv(ROOT / CFG.gold_path)
assert CFG.gold_rel_col in gold.columns
if CAND_COL not in gold.columns:
    alt = "cand_id" if CAND_COL == "user_id" else "user_id"
    gold = gold.rename(columns={alt: CAND_COL})

space = ADAPTER.prepare_feature_space(
    n_jobs=CFG.n_jobs_sample,
    n_candidates=CFG.n_cands_sample,
)
all_jobs = pd.read_csv(ADAPTER.job_path)
all_users = pd.read_csv(ADAPTER.user_path)
job_indices = gold["job_id"].map(ADAPTER.parse_id_index)
cand_indices = gold[CAND_COL].map(ADAPTER.parse_id_index)
assert job_indices.between(0, len(all_jobs) - 1).all(), "Graded JOB raw index outside source data"
assert cand_indices.between(0, len(all_users) - 1).all(), "Graded CV raw index outside source data"
expected_job_titles = job_indices.map(all_jobs["Job Title"])
expected_desired_jobs = cand_indices.map(all_users["Desired Job"])
assert gold["job_title"].fillna("").str.strip().equals(
    expected_job_titles.fillna("").str.strip()
), "Graded JOB raw IDs no longer match annotated job_title"
assert gold["desired_job"].fillna("").str.strip().equals(
    expected_desired_jobs.fillna("").str.strip()
), "Graded CV raw IDs no longer match annotated desired_job"
print("GRADED IDENTITY CHECK OK: raw JOB/CV IDs match annotated titles")

GOLD_JOBS = sorted(gold["job_id"].unique().tolist())
assert set(GOLD_JOBS).isdisjoint(set(df_raw["job_id"])), "Graded jobs leaked into sampled training pool"
print("Gold pairs", len(gold), "| jobs", len(GOLD_JOBS))
print(gold[CFG.gold_rel_col].value_counts().sort_index())
assert set(gold[CFG.gold_rel_col].unique()) <= {0, 1, 2, 3}

gold_feats = ADAPTER.features_for_pairs(
    gold, cand_col=CAND_COL,
    n_jobs=CFG.n_jobs_sample, n_candidates=CFG.n_cands_sample,
)
gold_feat = gold.merge(gold_feats, on=["job_id", CAND_COL], how="inner")
print("Gold feature coverage", len(gold_feat), "/", len(gold))
assert len(gold_feat) >= 0.95 * len(gold)

overlap = df_raw.merge(
    gold_feat[["job_id", CAND_COL]],
    on=["job_id", CAND_COL],
)
print("Overlap with sampled weak-label pairs", len(overlap), "/", len(gold_feat))
assert len(overlap) == 0, "Graded pairs leaked into sampled weak-label data"
assert np.isfinite(gold_feat[list(FEATURE_COLS)].to_numpy(float)).all()
print("FEATURE SPACE CHECK OK: graded raw rows use frozen paper-subsample TF-IDF transformers")

tr, va, te, meta = job_disjoint_split(df_raw, 42, GOLD_JOBS, CFG.train_ratio, CFG.val_ratio)
print("seed42 jobs train/val/test", len(meta["train_jobs"]), len(meta["val_jobs"]), len(meta["test_jobs"]))
assert set(GOLD_JOBS).isdisjoint(set(meta["train_jobs"]) | set(meta["val_jobs"]) | set(meta["test_jobs"]))
print("LEAKAGE CHECK OK: gold jobs excluded from all splits")
"""))

cells.append(md(r"""
## 06. Weak Label Analysis

LF → Dawid–Skene → `y_prob` (**weak relevance probability**, không phải GT).

Báo cáo coverage / conflict / pairwise agreement / correlation. Không dùng Fleiss’ κ như test độc lập.
"""))
cells.append(code(r"""
def lf_quality_report(df_lfs: pd.DataFrame, lf_cols=LF_COLS) -> dict:
    cols = [c for c in lf_cols if c in df_lfs.columns]
    out = {
        "coverage": {}, "positive_rate": {}, "negative_rate": {},
        "abstain_rate": {}, "conflict_rate": None, "pairwise_agree": {},
    }
    for c in cols:
        out["coverage"][c] = float((df_lfs[c] != 0).mean())
        out["positive_rate"][c] = float((df_lfs[c] == 1).mean())
        out["negative_rate"][c] = float((df_lfs[c] == -1).mean())
        out["abstain_rate"][c] = float((df_lfs[c] == 0).mean())
    mat = df_lfs[cols].values
    conflicts = active = 0
    for row in mat:
        nz = row[row != 0]
        if len(nz) >= 2:
            active += 1
            if nz.min() < 0 < nz.max():
                conflicts += 1
    out["conflict_rate"] = float(conflicts / active) if active else float("nan")
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            mask = (df_lfs[a] != 0) & (df_lfs[b] != 0)
            out["pairwise_agree"][f"{a}|{b}"] = (
                float("nan") if mask.sum() == 0
                else float((df_lfs.loc[mask, a] == df_lfs.loc[mask, b]).mean())
            )
    return out

# Demo LF stats on seed-42 split only (analysis). Full training happens in §14.
pipe0 = WeakLabelPipeline(CFG.lf_pos_percentile, CFG.lf_neg_percentile)
train0_w = pipe0.fit_transform_train(tr, method=CFG.aggregator)
pipe0_parameters = pipe0.aggregator_parameters()
gold_w = pipe0.transform(gold_feat, method=CFG.aggregator).copy()
assert pipe0.aggregator_parameters() == pipe0_parameters
gold_w[CFG.gold_rel_col] = gold_feat[CFG.gold_rel_col].values

q = lf_quality_report(train0_w)
consensus0 = pipe0.aggregator_consensus.fit(train0_w, lf_cols=LF_COLS).predict_proba(train0_w)
params0 = pipe0.aggregator_parameters()
bounds0 = np.asarray(params0["source_sensitivities"] + params0["source_specificities"])
boundary_rate0 = float(np.mean(np.isclose(bounds0, 0.55) | np.isclose(bounds0, 0.95)))
prior_at_boundary0 = bool(np.isclose(params0["p_prior"], 0.05) | np.isclose(params0["p_prior"], 0.95))
print("y_prob mean/std", float(train0_w["y_prob"].mean()), float(train0_w["y_prob"].std()))
print("y_prob quantiles", train0_w["y_prob"].quantile([0, .25, .5, .75, 1]).to_dict())
print(json.dumps(q, indent=2))
print({
    "ds_consensus_mae": float(np.mean(np.abs(train0_w["y_prob"].values - consensus0))),
    "parameter_boundary_rate": boundary_rate0,
    "prior_at_boundary": prior_at_boundary0,
})
if prior_at_boundary0 or boundary_rate0 >= 0.5:
    print("WARNING: Dawid-Skene parameters are boundary-heavy; treat weak probabilities cautiously.")
display(pipe0.lf_correlation(train0_w).round(3))
print("NOTE: high LF correlation is expected (shared features) — not independence evidence.")
"""))

cells.append(code(r"""
def eval_ndcg_map(df, scores, rel_col, k_list=(5, 10)):
    tmp = df.copy(); tmp["_s"] = scores
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


class Predictor:
    def __init__(self, name, fn):
        self.name = name
        self._fn = fn
    def predict(self, df):
        return self._fn(df)
"""))

cells.append(md(r"""
## 07. Model H — Hand-crafted baseline

`Score = 0.30 loc + 0.25 skill + 0.20 exp + 0.15 role + 0.10 desc` — không train.
"""))
cells.append(code(r"""
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

print("H gold (seed42 LF pipeline demo)", eval_ndcg_map(gold_w, ModelH().predict(gold_w), CFG.gold_rel_col))
"""))

cells.append(md(r"""
## 08. Model B1 — Learned linear weighting (ablation)

LogisticRegression trên `x`, target = `heuristic_label` (cắt từ H). Chỉ ablation.
"""))
cells.append(code(r"""
class ModelB1:
    name = "B1"
    def __init__(self, label_col="heuristic_label"):
        self.label_col = label_col
        self.clf = LogisticRegression(C=1.0, max_iter=500, random_state=0)
    def fit(self, df_train):
        self.clf.fit(df_train[list(CFG.feature_cols)].values, df_train[self.label_col].values.astype(int))
        return self
    def predict(self, df):
        return self.clf.predict_proba(df[list(CFG.feature_cols)].values)[:, 1]
    @property
    def weights_(self):
        return {f: float(w) for f, w in zip(CFG.feature_cols, self.clf.coef_.ravel())}

b1 = ModelB1().fit(train0_w)
print("B1 weights", b1.weights_)
print("B1 gold (demo)", eval_ndcg_map(gold_w, b1.predict(gold_w), CFG.gold_rel_col))
"""))

cells.append(md(r"""
## 09. Model B2 — Pointwise soft weak label (ablation)

Soft target `y_prob` + **BCEWithLogitsLoss** (linear score).
"""))
cells.append(code(r"""
class ModelB2:
    name = "B2"
    def __init__(self, target_col="y_prob", lr=1e-2, epochs=200, seed=0):
        self.target_col = target_col
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.lin = None

    def fit(self, df_train):
        set_seed(self.seed)
        X = torch.tensor(df_train[list(CFG.feature_cols)].values, dtype=torch.float32)
        y = torch.tensor(df_train[self.target_col].values, dtype=torch.float32)
        self.lin = nn.Linear(X.shape[1], 1)
        opt = optim.Adam(self.lin.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(self.lin(X).squeeze(-1), y)
            loss.backward(); opt.step()
        return self

    def predict(self, df):
        self.lin.eval()
        X = torch.tensor(df[list(CFG.feature_cols)].values, dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(self.lin(X).squeeze(-1)).cpu().numpy()

b2 = ModelB2().fit(train0_w)
print("B2 gold (demo)", eval_ndcg_map(gold_w, b2.predict(gold_w), CFG.gold_rel_col))
"""))

cells.append(md(r"""
## 10. RankNet Pair Generation

Per JD: `(i,j)` nếu `y_prob_i - y_prob_j > margin`. Trần `max_pairs_per_job`.
"""))
cells.append(code(r"""
Xi, Xj = build_rank_pairs(
    train0_w, score_col="y_prob", max_pairs_per_job=CFG.max_pairs_per_job,
    margin=CFG.pair_margin, rng=np.random.RandomState(42),
)
print("pairs", len(Xi), "dim", Xi.shape[1] if len(Xi) else 0)
"""))

cells.append(md(r"""
## 11. Model M1 — RankNet (RQ1)

Arch: `5 → 32 → ReLU → Dropout → 16 → ReLU → 1`.
`L_rank = softplus(-(s_A - s_B))`.

**Early-stop & HP:** minimize **val pair-loss** (không NDCG y_prob, không gold).

**Seed kép:** mỗi lần `train_m1` gọi `set_seed(seed)` → mọi `(lr, bs)` trong grid cùng init; so sánh HP within-seed công bằng. Variance across seeds (§14) gồm cả split lẫn init.
"""))
cells.append(code(r"""
class RankNetMLP(nn.Module):
    def __init__(self, input_dim=None, h1=32, h2=16, dropout=0.1):
        super().__init__()
        input_dim = len(CFG.feature_cols) if input_dim is None else input_dim
        assert input_dim == len(EXPECTED_FEATURE_COLS)
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h = self.dropout(F.relu(self.fc1(x)))
        h = F.relu(self.fc2(h))
        return self.out(h).squeeze(-1)
    def hidden(self, x):
        h = self.dropout(F.relu(self.fc1(x)))
        return F.relu(self.fc2(h))


def _batches(n, bs, rng):
    idx = np.arange(n); rng.shuffle(idx)
    for s in range(0, n, bs):
        yield idx[s:s + bs]


def predict_net(net, df):
    net.eval()
    X = torch.tensor(df[list(CFG.feature_cols)].values, dtype=torch.float32)
    with torch.no_grad():
        return net(X).cpu().numpy()


def pair_tensors(df, seed, cfg=CFG):
    Xi, Xj, used_m = build_rank_pairs_robust(
        df, score_col="y_prob", max_pairs_per_job=cfg.max_pairs_per_job,
        margin=cfg.pair_margin, margin_fallbacks=(0.01, 0.0), min_pairs=8,
        rng=np.random.RandomState(seed),
    )
    return torch.tensor(Xi, dtype=torch.float32), torch.tensor(Xj, dtype=torch.float32), float(used_m)


def weak_score_audit(df, split, seed):
    group_sizes = df.groupby("job_id")["y_prob"].size()
    unique_counts = df.groupby("job_id")["y_prob"].nunique()
    tied_pairs = total_pairs = 0
    for _, group in df.groupby("job_id"):
        counts = group["y_prob"].value_counts().to_numpy(dtype=int)
        tied_pairs += int(np.sum(counts * (counts - 1) // 2))
        n = len(group)
        total_pairs += n * (n - 1) // 2
    return {
        "seed": seed,
        "split": split,
        "n_rows": len(df),
        "n_jobs": int(len(group_sizes)),
        "n_unique_scores": int(df["y_prob"].nunique()),
        "median_unique_scores_per_job": float(unique_counts.median()),
        "tie_pair_rate": float(tied_pairs / total_pairs) if total_pairs else float("nan"),
    }


def eval_pair_loss(net, ti, tj):
    if len(ti) == 0:
        return float("nan")
    net.eval()
    with torch.no_grad():
        return float(F.softplus(-(net(ti) - net(tj))).mean().item())


@dataclass
class TrainResult:
    model_name: str
    seed: int
    best_epoch: int
    best_val_loss: float
    lr: float
    batch_size: int
    lambda_gap: float
    train_seconds: float
    monitor_source: str = "val"
    val_margin_used: float = float("nan")
    n_monitor_pairs: int = 0
    train_margin_used: float = float("nan")
    n_train_pairs: int = 0
    gap_vocab_size: int = 0
    gap_target_prevalence: float = float("nan")
    selected_checkpoint_gap_loss: float = float("nan")
    last_epoch_train_gap_loss: float = float("nan")
    stopped_early: bool = False
    reached_max_epochs: bool = False
    net: Optional[nn.Module] = None
    gap_head: Optional[nn.Module] = None


def _require_validation_pairs(vi, vj, val_m, seed, model_name):
    if len(vi) == 0:
        raise RuntimeError(
            f"{model_name} seed={seed}: validation produced no rank pairs even at margin=0. "
            "Cannot perform independent early stopping or hyperparameter selection."
        )
    print(f"  val_pairs={len(vi)} margin_used={val_m}")
    return vi, vj


def train_m1(df_train, df_val, seed, lr, batch_size, cfg=CFG) -> TrainResult:
    rng = set_seed(seed)
    ti, tj, train_m = pair_tensors(df_train, seed, cfg)
    vi, vj, val_m = pair_tensors(df_val, seed + 1, cfg)
    net = RankNetMLP(len(cfg.feature_cols), cfg.hidden1, cfg.hidden2, cfg.dropout)
    if len(ti) == 0:
        raise RuntimeError(f"M1 seed={seed}: training produced no rank pairs")
    monitor_i, monitor_j = _require_validation_pairs(vi, vj, val_m, seed, "M1")
    opt = optim.Adam(net.parameters(), lr=lr)
    best_state, best_loss, best_ep, wait = None, float("inf"), 0, 0
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        net.train()
        for b in _batches(len(ti), batch_size, rng):
            opt.zero_grad()
            F.softplus(-(net(ti[b]) - net(tj[b]))).mean().backward()
            opt.step()
        vloss = eval_pair_loss(net, monitor_i, monitor_j)
        if not np.isfinite(vloss):
            wait += 1
            if wait >= cfg.patience:
                break
            continue
        if vloss < best_loss - 1e-5:
            best_loss, best_ep, wait = vloss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError(f"M1 seed={seed}: no finite validation checkpoint was produced")
    net.load_state_dict(best_state)
    reached_max_epochs = epoch == cfg.max_epochs
    return TrainResult(
        "M1", seed, best_ep, best_loss, lr, batch_size, 0.0, time.time() - t0,
        monitor_source="val", val_margin_used=val_m, n_monitor_pairs=len(monitor_i),
        train_margin_used=train_m, n_train_pairs=len(ti),
        stopped_early=not reached_max_epochs, reached_max_epochs=reached_max_epochs,
        net=net,
    )


def select_hp_m1(df_train, df_val, seed, cfg=CFG):
    best = (float("inf"), cfg.default_lr, cfg.default_batch)
    for lr in cfg.lr_grid:
        for bs in cfg.batch_grid:
            tr = train_m1(df_train, df_val, seed, lr, bs, cfg)
            if tr.best_val_loss < best[0]:
                best = (tr.best_val_loss, lr, bs)
    if not np.isfinite(best[0]):
        raise RuntimeError(f"M1 seed={seed}: hyperparameter selection has no finite validation loss")
    return best[1], best[2]

print(RankNetMLP())
"""))

cells.append(md(r"""
## 12–13. Skill-gap & Model M2 (RQ2)

`L = L_rank + λ L_gap` khi train.

**λ tune trên val pair-loss thuần (rank loss only, không gồm gap loss)** — tránh λ tự tối ưu theo auxiliary objective của chính nó.
"""))
cells.append(code(r"""
def train_m2(df_train, df_val, seed, lr, batch_size, lambda_gap, cfg=CFG) -> TrainResult:
    rng = set_seed(seed)
    ti, tj, train_m = pair_tensors(df_train, seed, cfg)
    vi, vj, val_m = pair_tensors(df_val, seed + 1, cfg)
    Y_gap, vocab = build_gap_targets(df_train)
    net = RankNetMLP(len(cfg.feature_cols), cfg.hidden1, cfg.hidden2, cfg.dropout)
    gap_head = SkillGapHead(cfg.hidden2, Y_gap.shape[1])
    if len(ti) == 0:
        raise RuntimeError(f"M2 seed={seed}: training produced no rank pairs")
    monitor_i, monitor_j = _require_validation_pairs(vi, vj, val_m, seed, "M2")
    X_all = torch.tensor(df_train[list(cfg.feature_cols)].values, dtype=torch.float32)
    Y_all = torch.tensor(Y_gap, dtype=torch.float32)
    opt = optim.Adam(list(net.parameters()) + list(gap_head.parameters()), lr=lr)
    best_state, best_gap, best_loss, best_ep, wait = None, None, float("inf"), 0, 0
    selected_checkpoint_gap_loss = last_epoch_train_gap_loss = float("nan")
    gap_target_prevalence = float(Y_gap.mean()) if Y_gap.size else float("nan")
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        net.train(); gap_head.train()
        for b in _batches(len(ti), batch_size, rng):
            opt.zero_grad()
            l_rank = F.softplus(-(net(ti[b]) - net(tj[b]))).mean()
            l_gap = skill_gap_loss(gap_head(net.hidden(X_all)), Y_all)
            (l_rank + lambda_gap * l_gap).backward()
            opt.step()
        net.eval(); gap_head.eval()
        with torch.no_grad():
            last_epoch_train_gap_loss = float(
                skill_gap_loss(gap_head(net.hidden(X_all)), Y_all).item()
            )
        vloss = eval_pair_loss(net, monitor_i, monitor_j)
        if not np.isfinite(vloss):
            wait += 1
            if wait >= cfg.patience:
                break
            continue
        if vloss < best_loss - 1e-5:
            best_loss, best_ep, wait = vloss, epoch, 0
            selected_checkpoint_gap_loss = last_epoch_train_gap_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            best_gap = {k: v.detach().cpu().clone() for k, v in gap_head.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError(f"M2 seed={seed}: no finite validation checkpoint was produced")
    net.load_state_dict(best_state)
    gap_head.load_state_dict(best_gap)
    reached_max_epochs = epoch == cfg.max_epochs
    return TrainResult(
        "M2", seed, best_ep, best_loss, lr, batch_size, lambda_gap, time.time() - t0,
        monitor_source="val", val_margin_used=val_m, n_monitor_pairs=len(monitor_i),
        train_margin_used=train_m, n_train_pairs=len(ti), gap_vocab_size=len(vocab),
        gap_target_prevalence=gap_target_prevalence,
        selected_checkpoint_gap_loss=selected_checkpoint_gap_loss,
        last_epoch_train_gap_loss=last_epoch_train_gap_loss,
        stopped_early=not reached_max_epochs, reached_max_epochs=reached_max_epochs,
        net=net, gap_head=gap_head,
    )


def select_lambda_m2(df_train, df_val, seed, lr, batch_size, cfg=CFG):
    best = (float("inf"), cfg.default_lambda_gap)
    for lam in cfg.lambda_gap_grid:
        tr = train_m2(df_train, df_val, seed, lr, batch_size, lam, cfg)
        if tr.best_val_loss < best[0]:
            best = (tr.best_val_loss, lam)
    if not np.isfinite(best[0]):
        raise RuntimeError(f"M2 seed={seed}: lambda selection has no finite validation loss")
    return best[1]

gap_probe = "python .. java ... .._• sql"
assert parse_skill_set(gap_probe) == {"python", "java", "sql"}
Y_gap_demo, gap_vocab_demo = build_gap_targets(train0_w)
assert all(parse_skill_set(token) == {token} for token in gap_vocab_demo)
print(
    "gap audit",
    {
        "vocab_size": len(gap_vocab_demo),
        "target_prevalence": float(Y_gap_demo.mean()) if Y_gap_demo.size else float("nan"),
        "target_sparsity": float(1.0 - Y_gap_demo.mean()) if Y_gap_demo.size else float("nan"),
        "sanitized_probe": sorted(parse_skill_set(gap_probe)),
    },
)
"""))

cells.append(md(r"""
## 14. Multi-seed Experiment Runner

Mỗi seed: job-disjoint (gold hold-out) → LF fit train-only → H/B1/B2/M1/M2 → đánh giá **chỉ trên gold**.
"""))
cells.append(code(r"""
def run_one_seed(seed, cfg=CFG):
    print(f"\n======== SEED {seed} ========")
    set_seed(seed)
    df_train, df_val, df_test, meta = job_disjoint_split(
        df_raw, seed, GOLD_JOBS, cfg.train_ratio, cfg.val_ratio)
    assert set(GOLD_JOBS).isdisjoint(set(meta["train_jobs"]) | set(meta["val_jobs"]) | set(meta["test_jobs"]))

    pipe = WeakLabelPipeline(cfg.lf_pos_percentile, cfg.lf_neg_percentile)
    df_train_w = pipe.fit_transform_train(df_train, method=cfg.aggregator)
    frozen_parameters = pipe.aggregator_parameters()
    df_val_w = pipe.transform(df_val, method=cfg.aggregator)
    assert pipe.aggregator_parameters() == frozen_parameters
    df_gold_w = pipe.transform(gold_feat, method=cfg.aggregator).copy()
    assert pipe.aggregator_parameters() == frozen_parameters
    df_gold_w[cfg.gold_rel_col] = gold_feat[cfg.gold_rel_col].values
    consensus = pipe.aggregator_consensus.fit(df_train_w, lf_cols=LF_COLS).predict_proba(df_train_w)
    parameter_values = np.asarray(
        frozen_parameters["source_sensitivities"] + frozen_parameters["source_specificities"]
    )
    boundary_rate = float(
        np.mean(np.isclose(parameter_values, 0.55) | np.isclose(parameter_values, 0.95))
    )
    label_model_audit = {
        "parameter_boundary_rate": boundary_rate,
        "prior_at_boundary": bool(
            np.isclose(frozen_parameters["p_prior"], 0.05)
            or np.isclose(frozen_parameters["p_prior"], 0.95)
        ),
        "ds_consensus_mae_train": float(
            np.mean(np.abs(df_train_w["y_prob"].values - consensus))
        ),
    }
    pair_audit = [
        weak_score_audit(df_train_w, "train", seed),
        weak_score_audit(df_val_w, "validation", seed),
        weak_score_audit(df_gold_w, "graded_eval", seed),
    ]
    print("  frozen_label_model", json.dumps({**frozen_parameters, **label_model_audit}))
    if label_model_audit["prior_at_boundary"] or boundary_rate >= 0.5:
        print("  WARNING: Dawid-Skene parameters are boundary-heavy for this seed.")
    print("  weak_score_audit", json.dumps(pair_audit))

    rows, models, b1_w = [], {}, None

    t0 = time.time(); mh = ModelH().fit(df_train_w)
    met = eval_ndcg_map(df_gold_w, mh.predict(df_gold_w), cfg.gold_rel_col, cfg.k_list)
    rows.append({"model": "H", "seed": seed, **met, "training_time": time.time() - t0,
                 "lr": None, "batch_size": None, "lambda_gap": None, "best_epoch": 0,
                 "best_val_loss": None})
    models["H"] = Predictor("H", mh.predict)

    t0 = time.time(); mb1 = ModelB1().fit(df_train_w)
    met = eval_ndcg_map(df_gold_w, mb1.predict(df_gold_w), cfg.gold_rel_col, cfg.k_list)
    rows.append({"model": "B1", "seed": seed, **met, "training_time": time.time() - t0,
                 "lr": None, "batch_size": None, "lambda_gap": None, "best_epoch": 0,
                 "best_val_loss": None})
    models["B1"] = Predictor("B1", mb1.predict)
    b1_w = {"seed": seed, **mb1.weights_}

    t0 = time.time(); mb2 = ModelB2(seed=seed).fit(df_train_w)
    met = eval_ndcg_map(df_gold_w, mb2.predict(df_gold_w), cfg.gold_rel_col, cfg.k_list)
    rows.append({"model": "B2", "seed": seed, **met, "training_time": time.time() - t0,
                 "lr": None, "batch_size": None, "lambda_gap": None, "best_epoch": 0,
                 "best_val_loss": None})
    models["B2"] = Predictor("B2", mb2.predict)

    lr, bs = select_hp_m1(df_train_w, df_val_w, seed, cfg)
    print(f"  HP lr={lr} bs={bs}")
    tr1 = train_m1(df_train_w, df_val_w, seed, lr, bs, cfg)
    met = eval_ndcg_map(df_gold_w, predict_net(tr1.net, df_gold_w), cfg.gold_rel_col, cfg.k_list)
    rows.append({"model": "M1", "seed": seed, **met, "training_time": tr1.train_seconds,
                 "lr": lr, "batch_size": bs, "lambda_gap": 0.0, "best_epoch": tr1.best_epoch,
                 "best_val_loss": tr1.best_val_loss, "monitor_source": tr1.monitor_source,
                 "train_margin_used": tr1.train_margin_used, "n_train_pairs": tr1.n_train_pairs,
                 "val_margin_used": tr1.val_margin_used, "n_monitor_pairs": tr1.n_monitor_pairs,
                 "stopped_early": tr1.stopped_early,
                 "reached_max_epochs": tr1.reached_max_epochs})
    models["M1"] = Predictor("M1", lambda d, n=tr1.net: predict_net(n, d))

    lam = select_lambda_m2(df_train_w, df_val_w, seed, lr, bs, cfg)
    print(f"  lambda_gap={lam}")
    tr2 = train_m2(df_train_w, df_val_w, seed, lr, bs, lam, cfg)
    met = eval_ndcg_map(df_gold_w, predict_net(tr2.net, df_gold_w), cfg.gold_rel_col, cfg.k_list)
    rows.append({"model": "M2", "seed": seed, **met, "training_time": tr2.train_seconds,
                 "lr": lr, "batch_size": bs, "lambda_gap": lam, "best_epoch": tr2.best_epoch,
                 "best_val_loss": tr2.best_val_loss, "monitor_source": tr2.monitor_source,
                 "train_margin_used": tr2.train_margin_used, "n_train_pairs": tr2.n_train_pairs,
                 "val_margin_used": tr2.val_margin_used, "n_monitor_pairs": tr2.n_monitor_pairs,
                 "gap_vocab_size": tr2.gap_vocab_size,
                 "gap_target_prevalence": tr2.gap_target_prevalence,
                 "selected_checkpoint_gap_loss": tr2.selected_checkpoint_gap_loss,
                 "last_epoch_train_gap_loss": tr2.last_epoch_train_gap_loss,
                 "stopped_early": tr2.stopped_early,
                 "reached_max_epochs": tr2.reached_max_epochs})
    models["M2"] = Predictor("M2", lambda d, n=tr2.net: predict_net(n, d))

    # optional checkpoint for representative reproducibility
    ckpt_dir = OUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tr1.net.state_dict(), ckpt_dir / f"m1_seed{seed}.pt")
    torch.save(tr2.net.state_dict(), ckpt_dir / f"m2_seed{seed}.pt")

    qualsens = {
        "M1": qualify_sensitivity(df_gold_w, models["M1"].predict),
        "M2": qualify_sensitivity(df_gold_w, models["M2"].predict),
    }
    return {
        "rows": rows, "models": models, "df_gold": df_gold_w,
        "qualsens": qualsens, "b1_weights": b1_w,
        "label_model_parameters": {
            "seed": seed, **frozen_parameters, **label_model_audit,
        },
        "pair_audit": pair_audit,
    }


all_rows, qualsens_by_seed, b1_weight_rows, bundles_by_seed = [], {}, [], {}
label_model_rows, pair_audit_rows = [], []
for sd in CFG.seeds:
    bundle = run_one_seed(sd, CFG)
    all_rows.extend(bundle["rows"])
    qualsens_by_seed[sd] = bundle["qualsens"]
    b1_weight_rows.append(bundle["b1_weights"])
    label_model_rows.append(bundle["label_model_parameters"])
    pair_audit_rows.extend(bundle["pair_audit"])
    bundles_by_seed[sd] = bundle

results_df = pd.DataFrame(all_rows)
results_df.to_csv(OUT_DIR / "results_all_runs.csv", index=False)
pd.DataFrame(b1_weight_rows).to_csv(OUT_DIR / "b1_weights_by_seed.csv", index=False)
(OUT_DIR / "label_model_parameters_by_seed.json").write_text(
    json.dumps(label_model_rows, indent=2), encoding="utf-8"
)
pd.DataFrame(pair_audit_rows).to_csv(OUT_DIR / "weak_score_pair_audit_by_seed.csv", index=False)
train_audit_cols = [
    "model", "seed", "best_epoch", "best_val_loss", "monitor_source",
    "train_margin_used", "n_train_pairs", "val_margin_used", "n_monitor_pairs",
    "stopped_early", "reached_max_epochs", "gap_vocab_size", "gap_target_prevalence",
    "selected_checkpoint_gap_loss", "last_epoch_train_gap_loss",
]
print("[AUDIT] M1/M2 model-selection monitors")
display(results_df.loc[results_df["model"].isin(["M1", "M2"]), train_audit_cols])
assert results_df.loc[results_df["model"].isin(["M1", "M2"]), "monitor_source"].eq("val").all()
display(results_df[["model", "seed", "ndcg@5", "ndcg@10", "map@5", "map@10", "best_epoch", "best_val_loss"]])
"""))

cells.append(code(r"""
order = ["H", "B1", "B2", "M1", "M2"]
metric_cols = [c for c in ["ndcg@5", "ndcg@10", "map@5", "map@10"] if c in results_df.columns]
summary_rows = []
for model, g in results_df.groupby("model", sort=False):
    row = {"model": model, "n_seeds": len(g)}
    for m in metric_cols:
        row[f"{m}_mean"] = g[m].mean()
        row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else 0.0
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
summary_df["model"] = pd.Categorical(summary_df["model"], order, ordered=True)
summary_df = summary_df.sort_values("model")
summary_df.to_csv(OUT_DIR / "results_summary_mean_std.csv", index=False)
display(summary_df)
"""))

cells.append(md(r"""
## 15. Statistical Analysis

**CI chính = seed-level paired deltas** (một Δ nDCG@5 mỗi seed, bootstrap trên 5 điểm).

| So sánh | RQ |
|---|---|
| B1 − H | ablation learned weighting |
| B1 − B2 | soft vs hard weak labels |
| M1 − B2, M1 − H | RQ1 |
| M2 − M1 | RQ2 |

Job-level paired bootstrap trong **từng** seed = diagnostic (sampling uncertainty của gold), không thay CI chính.

**Quy tắc:** nếu seed-level 95% CI chứa 0 → không kết luận cải thiện chắc chắn. n_seeds=5 → CI rộng; gold ~12 JD → power thấp.
"""))
cells.append(code(r"""
def seed_level_bootstrap_ci(deltas, n_boot=1000, seed=42):
    deltas = np.asarray(deltas, dtype=float)
    rng = np.random.RandomState(seed)
    if len(deltas) == 0:
        return {"mean_delta": float("nan"), "ci_95_low": float("nan"),
                "ci_95_high": float("nan"), "n_seeds": 0}
    boots = [float(np.mean(rng.choice(deltas, size=len(deltas), replace=True)))
             for _ in range(n_boot)]
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_95_low": float(np.percentile(boots, 2.5)),
        "ci_95_high": float(np.percentile(boots, 97.5)),
        "n_seeds": int(len(deltas)),
        "deltas": deltas.tolist(),
    }


comparisons = [
    ("B1_vs_H", "H", "B1"),
    ("B2_vs_B1", "B1", "B2"),  # soft vs hard weak labels
    ("M1_vs_H", "H", "M1"),
    ("M1_vs_B2", "B2", "M1"),
    ("M2_vs_M1", "M1", "M2"),
]

pivot = results_df.pivot(index="seed", columns="model", values="ndcg@5")
ci_rows = []
for name, a, b in comparisons:
    deltas = (pivot[b] - pivot[a]).reindex(list(CFG.seeds)).values
    res = seed_level_bootstrap_ci(deltas, n_boot=CFG.n_bootstrap, seed=CFG.seeds[0])
    excludes = bool(res["ci_95_low"] > 0 or res["ci_95_high"] < 0)
    ci_rows.append({
        "comparison": name,
        "metric": "ndcg@5",
        "level": "seed",
        "mean_delta": res["mean_delta"],
        "ci_95_low": res["ci_95_low"],
        "ci_95_high": res["ci_95_high"],
        "n_seeds": res["n_seeds"],
        "ci_excludes_zero": excludes,
        "interpretation": "CI excludes 0" if excludes else "CI contains 0 — no strong claim",
        "deltas_by_seed": json.dumps({str(s): float(d) for s, d in zip(CFG.seeds, res["deltas"])}),
    })
ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv(OUT_DIR / "confidence_interval_seed_level.csv", index=False)
display(ci_df[["comparison", "mean_delta", "ci_95_low", "ci_95_high", "ci_excludes_zero", "interpretation"]])

# Diagnostic: job-bootstrap per seed (not primary claim)
diag_rows = []
for sd, bundle in bundles_by_seed.items():
    df_g = bundle["df_gold"]
    models = bundle["models"]
    for name, a, b in [("M1_vs_H", "H", "M1"), ("M1_vs_B2", "B2", "M1"),
                       ("B2_vs_B1", "B1", "B2"), ("M2_vs_M1", "M1", "M2")]:
        res = paired_bootstrap_test(
            df_g, models[a], models[b],
            metric_fn=ndcg_at_k, k=5, target_col=CFG.gold_rel_col,
            n_bootstraps=CFG.n_bootstrap, seed=sd,
        )
        diag_rows.append({
            "seed": sd, "comparison": name, "level": "job_within_seed",
            "mean_delta": res["mean_delta"],
            "ci_95_low": res["ci_95_low"], "ci_95_high": res["ci_95_high"],
            "p_value": res["p_value"],
            "ci_excludes_zero": bool(res["ci_95_low"] > 0 or res["ci_95_high"] < 0),
        })
diag_df = pd.DataFrame(diag_rows)
diag_df.to_csv(OUT_DIR / "confidence_interval_job_within_seed.csv", index=False)
print("Diagnostic job-bootstrap (not primary):")
display(diag_df.groupby("comparison")[["mean_delta", "ci_excludes_zero"]].mean())

(OUT_DIR / "rq2_qualsens_by_seed.json").write_text(
    json.dumps({str(k): v for k, v in qualsens_by_seed.items()}, indent=2), encoding="utf-8")
print("QualSens seed0:", json.dumps(qualsens_by_seed[CFG.seeds[0]], indent=2))
"""))

cells.append(md("## 16. Visualization"))
cells.append(code(r"""
plot_df = results_df.copy()
plot_df["model"] = pd.Categorical(plot_df["model"], order, ordered=True)

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(data=plot_df, x="model", y="ndcg@5", errorbar="sd", ax=ax, color="#4C72B0")
ax.set_title("Graded eval set — nDCG@5 (mean ± std across seeds)")
ax.set_ylim(0, min(1.05, plot_df["ndcg@5"].max() + 0.15))
fig.tight_layout(); fig.savefig(FIG_DIR / "fig1_model_comparison_ndcg5.png", dpi=200); plt.show()

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=plot_df, x="model", y="ndcg@5", ax=ax, color="#55A868")
sns.stripplot(data=plot_df, x="model", y="ndcg@5", ax=ax, color="black", size=6, alpha=0.7)
ax.set_title("Seed variance — nDCG@5")
fig.tight_layout(); fig.savefig(FIG_DIR / "fig2_seed_variance.png", dpi=200); plt.show()

abl = plot_df[plot_df["model"].isin(["M1", "M2"])]
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.barplot(data=abl, x="model", y="ndcg@5", hue="model", errorbar="sd", ax=ax, legend=False)
ax.set_title("RQ2 — M1 vs M2 (nDCG@5)")
fig.tight_layout(); fig.savefig(FIG_DIR / "fig3_m1_vs_m2.png", dpi=200); plt.show()
print("Figures →", FIG_DIR)
"""))

cells.append(md(r"""
## 17. Research Interpretation (bắt buộc điền sau Run All)

### RQ1 — LTR có cải thiện ranking trên graded set?

Dùng **seed-level CI** (`confidence_interval_seed_level.csv`), không dùng job-bootstrap 1 seed làm claim chính.

| So sánh | Δ nDCG@5 | 95% CI (seed) | CI loại 0? | Kết luận |
|---|---|---|---|---|
| B1 − H | | | | ablation |
| B2 − B1 | | | | soft vs hard |
| M1 − B2 | | | | |
| M1 − H | | | | |

Ổn định qua seeds? (std / boxplot)

### RQ2 — skill-gap

| | M1 QualSens | M2 QualSens |
|---|---|---|
| skill | | |
| exp | | |
| domain | | |

Nếu M2 ↑ NDCG nhưng QualSens không ↑: **skill-gap cải thiện ranking score nhưng chưa chứng minh phản ánh năng lực tốt hơn.**

### Giới hạn

1. Không hiring outcome — `y_prob` = weak relevance; gold ≠ GT chuyên gia.
2. LF phụ thuộc cùng 5 feature — không claim nguồn nhãn ngoài feature.
3. B1 target cắt từ H — không overclaim learned weighting.
4. Gold map 0–3, 1 annotator, ~12 JD → CI rộng / power thấp.
5. RankNet = hàm xếp hạng, không phải bộ trọng số tối ưu.
6. QualSens = perturbation feature bảng, không phải thử nghiệm tuyển dụng thật.
7. Job-disjoint cho phép CV overlap giữa split — không claim generalization sang ứng viên hoàn toàn mới.
8. Seed-level CI chỉ có 5 điểm — trung thực với variance train, nhưng interval rộng.
9. Graded JD/CV dùng raw-row IDs của snapshot Phase 3 và nằm ngoài sampled weak-label pairs; feature vẫn dùng frozen TF–IDF transformers của paper subsample.

### Reviewer checklist

- [x] Job-disjoint + gold hold-out
- [x] LF fit train-only
- [x] HP/early-stop theo val `L_rank`
- [x] B2 = soft BCEWithLogits
- [x] Shared TF–IDF adapter train/gold + overlap assert
- [x] Multi-seed mean±std
- [x] **Primary CI = seed-level paired deltas** (không neo 1 seed)
- [x] Bootstrap job-within-seed chỉ diagnostic
- [x] Có B2_vs_B1 trong bảng so sánh (soft vs hard)
- [x] λ chọn theo val pair-loss thuần
- [x] Interpretation không vượt evidence
"""))

cells.append(code(r"""
print("Primary CI artifact:", OUT_DIR / "confidence_interval_seed_level.csv")
print("Artifacts:")
for p in sorted(OUT_DIR.rglob("*")):
    if p.is_file():
        print(" -", p.relative_to(ROOT))
print("Done.")
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {OUT} cells={len(cells)}")
