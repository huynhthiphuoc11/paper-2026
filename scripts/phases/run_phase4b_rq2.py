"""
Phase 4b — RQ2: M1 vs M2 + QualSens theo nhóm perturbation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data import load_dataset, CVJobDatasetLoader
from src.weak import WeakLabelPipeline
from src.models import M1_RankNet, M2_RankNetSkillGap
from src.eval import qualify_sensitivity

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "phase4b"


def main():
    raise RuntimeError(
        "Legacy/non-authoritative entry point. Run: python "
        "scripts/run_paper_experiment.py --config configs/experiment.yaml"
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    loader = CVJobDatasetLoader(random_seed=42)
    df_raw = load_dataset(data_dir=str(ROOT / "data"), random_seed=42)
    df_train, _, df_test = loader.get_job_disjoint_splits(df_raw)

    pipe = WeakLabelPipeline()
    df_train_w = pipe.fit_transform_train(df_train, method="dawid_skene")
    df_test_w = pipe.aggregate(pipe.transform_lfs(df_test), method="dawid_skene")

    m1 = M1_RankNet()
    m2 = M2_RankNetSkillGap(lambda_gap=0.2)
    print("[4b] Train M1 / M2...")
    m1.fit(df_train_w)
    m2.fit(df_train_w)

    print("[4b] QualSens on test features (isolated)...")
    report = {
        "M1": qualify_sensitivity(df_test_w, m1.predict),
        "M2": qualify_sensitivity(df_test_w, m2.predict),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    (REPORT_DIR / "rq2_qualsens.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[4b] Wrote {REPORT_DIR}")


if __name__ == "__main__":
    main()
