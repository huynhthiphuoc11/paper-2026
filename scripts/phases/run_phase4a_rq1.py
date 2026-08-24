"""
Phase 4a — RQ1: H → B1/B2 → M1 trên gold graded (job-disjoint).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_dataset, CVJobDatasetLoader, FEATURE_COLS
from src.weak import WeakLabelPipeline
from src.models import build_default_models
from src.eval import evaluate_models_on_dataset, paired_bootstrap_test

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "phase4a"
GOLD_PATH = ROOT / "data" / "gold" / "human_validated_benchmark_graded_0_3.csv"


def _ensure_y_prob(df: pd.DataFrame, pipe: WeakLabelPipeline) -> pd.DataFrame:
    out = pipe.aggregate(pipe.transform_lfs(df), method="dawid_skene")
    return out


def main():
    raise RuntimeError(
        "Legacy/non-authoritative entry point. Run: python "
        "scripts/run_paper_experiment.py --config configs/experiment.yaml"
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("[4a] Load data + job-disjoint splits...")
    loader = CVJobDatasetLoader(random_seed=42)
    df_raw = load_dataset(data_dir=str(ROOT / "data"), random_seed=42)
    df_train, df_dev, df_test = loader.get_job_disjoint_splits(df_raw)

    print("[4a] Fit weak labels on TRAIN only...")
    pipe = WeakLabelPipeline()
    df_train_w = pipe.fit_transform_train(df_train, method="dawid_skene")
    df_test_w = _ensure_y_prob(df_test, pipe)

    # Gold: nếu có file author-annotated, merge theo khóa cặp; không thì dùng test split.
    if GOLD_PATH.exists():
        gold = pd.read_csv(GOLD_PATH)
        print(f"[4a] Gold file: {GOLD_PATH} ({len(gold)} rows)")
        # Kỳ vọng có cột relevance graded; tên cột linh hoạt.
        # Ưu tiên cột graded 0–3 (`relevance`); không gọi ground truth.
        rel_col = next(
            (c for c in ["relevance", "human_relevance", "label", "grade"] if c in gold.columns),
            None,
        )
        eval_df = df_test_w
        target_col = "gold_relevance" if "gold_relevance" in eval_df.columns else None
        if rel_col and {"job_id", "user_id"}.issubset(gold.columns) and {"job_id", "user_id"}.issubset(df_test_w.columns):
            eval_df = df_test_w.merge(
                gold[["job_id", "user_id", rel_col]],
                on=["job_id", "user_id"],
                how="inner",
            )
            target_col = rel_col
            print(f"[4a] Merged gold pairs: {len(eval_df)}; target={target_col}")
        if target_col is None:
            raise RuntimeError("Không tìm thấy cột graded relevance cho đánh giá.")
    else:
        eval_df = df_test_w
        target_col = "gold_relevance"
        print("[4a] Không có gold file — fallback gold_relevance trên test split.")

    models = build_default_models()
    # Phase 4a: chưa chạy M2
    models.pop("M2", None)

    print("[4a] Train H/B1/B2/M1...")
    # H không fit
    models["B1"].fit(df_train_w)
    models["B2"].fit(df_train_w)
    models["M1"].fit(df_train_w)

    print("[4a] Evaluate...")
    # evaluate_models_on_dataset kỳ vọng dict model có .predict
    table = evaluate_models_on_dataset(models, eval_df, target_col=target_col)
    print(table.to_string(index=False))
    table.to_csv(REPORT_DIR / "rq1_metrics.csv", index=False)

    # Bootstrap: M1 vs H, M1 vs B2
    tests = {}
    for name, base_key in [("M1_vs_H", "H"), ("M1_vs_B2", "B2"), ("B1_vs_H", "H")]:
        try:
            tests[name] = paired_bootstrap_test(
                eval_df, models[base_key], models["M1" if "M1" in name else "B1"], k=10
            )
        except Exception as e:
            tests[name] = {"error": str(e)}

    # Serialize bootstrap
    serializable = {}
    for k, v in tests.items():
        if isinstance(v, dict):
            serializable[k] = {
                kk: (float(vv) if isinstance(vv, (float, np.floating)) else vv)
                for kk, vv in v.items()
                if kk != "deltas"
            }
    (REPORT_DIR / "rq1_bootstrap.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[4a] Wrote {REPORT_DIR}")


if __name__ == "__main__":
    main()
