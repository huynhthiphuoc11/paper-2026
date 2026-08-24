from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src.models.skill_gap import parse_skill_set

def audit_skill_gap(df: pd.DataFrame, label_col="y_prob") -> dict:
    gaps = [len(parse_skill_set(j) - parse_skill_set(c)) for j, c in zip(df.job_skills, df.user_skills)]
    s = pd.Series(gaps, name="missing_skill_count")
    return {"n": int(len(s)), "zero_gap_rate": float((s == 0).mean()), "mean_missing_skills": float(s.mean()),
            "variance": float(s.var(ddof=0)), "correlation_with_label": float(s.corr(df[label_col], method="spearman")) if label_col in df else None,
            "distribution": {str(k): int(v) for k, v in s.value_counts().sort_index().items()}}

def write_skill_gap_audit(df, output_dir, label_col="y_prob"):
    report = audit_skill_gap(df, label_col); path = Path(output_dir); path.mkdir(parents=True, exist_ok=True)
    (path / "skill_gap_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
