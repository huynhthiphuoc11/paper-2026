"""Run pre-model diagnostics on a training partition only.

Usage: python scripts/run_research_audit.py --output reports/research_audit
Gold labels are deliberately never loaded here.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.data.loader import RealKaggleDatasetAdapter, FEATURE_COLS
from src.feature_audit import write_feature_audit
from src.weak import WeakLabelPipeline
from src.weak.pipeline import LF_COLS
from src.weak_label_audit import write_weak_label_audit
from src.skill_gap_audit import write_skill_gap_audit
from src.skill_normalization import write_skill_taxonomy_v2

def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", default="reports/research_audit")
    p.add_argument("--jobs", type=int, default=80); p.add_argument("--candidates", type=int, default=120)
    args = p.parse_args(); out = Path(args.output)
    adapter = RealKaggleDatasetAdapter("data", random_seed=42, df_threshold=.15)
    df = adapter.load_and_preprocess(args.jobs, args.candidates)
    # The fixed first 80% of job IDs is an audit/train partition only. No gold is read.
    train_jobs = sorted(df.job_id.unique())[:int(.8 * df.job_id.nunique())]
    train = df[df.job_id.isin(train_jobs)].copy()
    write_feature_audit(train, FEATURE_COLS + ["experience_gap"], out)
    pipe = WeakLabelPipeline(); weak = pipe.fit_transform_train(train)
    write_weak_label_audit(weak, LF_COLS, out)
    write_skill_gap_audit(weak, out)
    space = adapter.prepare_feature_space(args.jobs, args.candidates)
    raw_values = list(space["df_jobs"]["Job Requirements"].fillna("")) + list(space["df_users"]["Skills"].fillna(""))
    # The taxonomy is an auditable mapping of extracted list entries, not a bag of words.
    import re
    entries = [part.strip() for value in raw_values for part in re.split(r"[,;|/\n]+", str(value)) if part.strip()]
    write_skill_taxonomy_v2(entries, out / "skill_taxonomy_v2.csv")
    print(f"Wrote audit artifacts to {out}")
if __name__ == "__main__": main()
