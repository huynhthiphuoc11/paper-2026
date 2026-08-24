"""Deterministic, auditable canonicalisation for CV--JD skill phrases."""
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

_ALIASES = {
    "python3": "python", "python 3": "python", "py": "python",
    "ml": "machine learning", "machine-learning": "machine learning",
    "dl": "deep learning", "powerbi": "power bi", "power-bi": "power bi",
    "sql server": "sql", "postgresql": "postgres", "nodejs": "node.js",
}
_KNOWN_PHRASES = sorted(set(_ALIASES.values()) | {
    "data analysis", "machine learning", "deep learning", "data science",
    "computer vision", "natural language processing", "power bi",
    "project management", "software engineering", "business analysis",
}, key=len, reverse=True)

def normalize_skill(skill: str) -> str:
    value = re.sub(r"\s+", " ", str(skill).strip().lower())
    value = re.sub(r"^[\W_]+|[\W_]+$", "", value)
    return _ALIASES.get(value, value)

def extract_normalized_skills(text: str) -> set[str]:
    """Extract comma/list-like skills while preserving known multi-word phrases.

    This deliberately avoids general word-tokenisation: unknown prose is not silently
    promoted to a "skill".  Extraction rules are deterministic and inspectable.
    """
    raw = str(text or "").lower()
    if raw in {"", "nan", "none"}:
        return set()
    found = {phrase for phrase in _KNOWN_PHRASES if re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", raw)}
    for part in re.split(r"[,;|/\n]+", raw):
        candidate = normalize_skill(part)
        if 2 <= len(candidate) <= 60 and candidate not in {"and", "with", "year", "years"}:
            found.add(candidate)
    return found

def build_skill_taxonomy_v2(values) -> pd.DataFrame:
    originals = sorted({str(v).strip() for v in values if str(v).strip()})
    return pd.DataFrame({"original_skill": originals,
                         "normalized_skill": [normalize_skill(v) for v in originals]})

def write_skill_taxonomy_v2(values, output_path: str | Path) -> pd.DataFrame:
    table = build_skill_taxonomy_v2(values)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table
