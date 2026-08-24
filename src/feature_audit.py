"""Feature diagnostics.  Reports describe data only; they never fit on gold labels."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

def feature_audit(df: pd.DataFrame, feature_cols: list[str], near_constant_threshold: float = .01) -> dict:
    report = {"n_rows": int(len(df)), "features": {}}
    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        counts = x.value_counts(dropna=False)
        report["features"][col] = {
            "missing": int(x.isna().sum()), "mean": float(x.mean()), "variance": float(x.var(ddof=0)),
            "unique_values": int(x.nunique(dropna=True)), "unique_value_ratio": float(x.nunique(dropna=True) / max(1, len(x))),
            "dominant_value": None if counts.empty else str(counts.index[0]),
            "dominant_value_ratio": 0.0 if counts.empty else float(counts.iloc[0] / len(x)),
            # A small absolute variance is normal for bounded similarities.  Flag
            # degeneracy only when it co-occurs with very low cardinality, or when
            # a single value dominates the data.
            "near_constant": bool(
                counts.iloc[0] / len(x) >= .90
                or (x.var(ddof=0) < near_constant_threshold and x.nunique(dropna=True) / max(1, len(x)) < .02)
            ),
            "quantiles": {str(q): float(x.quantile(q)) for q in (.0, .25, .5, .75, 1.0)},
        }
    return report

def write_feature_audit(df, feature_cols, output_dir) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    report = feature_audit(df, feature_cols)
    (output / "feature_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # CSV histogram bins are portable and more reviewable than an image-only artifact.
    rows = []
    for col in feature_cols:
        hist, edges = np.histogram(pd.to_numeric(df[col], errors="coerce").dropna(), bins=20)
        rows += [{"feature": col, "left": float(edges[i]), "right": float(edges[i+1]), "count": int(hist[i])} for i in range(len(hist))]
    pd.DataFrame(rows).to_csv(output / "feature_histograms.csv", index=False)
    return report
