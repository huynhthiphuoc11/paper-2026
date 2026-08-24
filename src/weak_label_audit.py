"""Weak-label diagnostics and pre-registered LF eligibility rules."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

def audit_weak_labels(df: pd.DataFrame, lf_cols: list[str]) -> dict:
    stats, conflict = {}, pd.DataFrame(0.0, index=lf_cols, columns=lf_cols)
    for col in lf_cols:
        v = df[col]
        coverage = float((v != 0).mean())
        stats[col] = {"coverage": coverage, "positive_rate": float((v == 1).mean()),
                      "negative_rate": float((v == -1).mean()), "abstain_rate": float((v == 0).mean()),
                      "eligible_for_ds": bool(coverage < 1.0 and v.nunique() > 1 and (v == 1).any() and (v == -1).any())}
    for a in lf_cols:
        for b in lf_cols:
            mask = (df[a] != 0) & (df[b] != 0)
            conflict.loc[a, b] = np.nan if not mask.any() else float((df.loc[mask, a] != df.loc[mask, b]).mean())
    return {"lf_stats": stats, "conflict_matrix": conflict.to_dict(),
            "correlation": df[lf_cols].corr(method="spearman").fillna(0).to_dict(),
            "eligible_lfs": [c for c, s in stats.items() if s["eligible_for_ds"]]}

def write_weak_label_audit(df, lf_cols, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True); report = audit_weak_labels(df, lf_cols)
    (out / "weak_label_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(report["lf_stats"]).T.to_csv(out / "lf_coverage.csv")
    pd.DataFrame(report["conflict_matrix"]).to_csv(out / "lf_conflict_matrix.csv")
    return report
