from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


JOB_REQUIRED_COLUMNS = [
    "JobID", "URL Job", "Job Title", "Job Description", "Job Requirements",
    "Job Address", "Years of Experience",
]
CANDIDATE_REQUIRED_COLUMNS = [
    "URL User", "UserID", "Desired Job", "Workplace Desired", "Target",
    "Skills", "Work Experience",
]
RAW_FILENAMES = {
    "jobs": "JOB_DATA_FINAL.csv",
    "candidates": "USER_DATA_FINAL.csv",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_data_paths(data_dir: str | Path) -> dict[str, Path]:
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Real data directory does not exist: {root}")
    paths = {name: root / filename for name, filename in RAW_FILENAMES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required real-data files are missing: " + ", ".join(missing)
        )
    return paths


def build_input_manifest(data_dir: str | Path) -> dict[str, dict]:
    paths = resolve_data_paths(data_dir)
    manifest = {}
    for name, path in paths.items():
        stat = path.stat()
        manifest[name] = {
            "path": str(path),
            "size_bytes": int(stat.st_size),
            "modified_at_utc": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "sha256": _sha256_file(path),
        }
    return manifest


def format_job_id(index: int) -> str:
    return f"JOB_{int(index)}"


def format_candidate_id(index: int) -> str:
    return f"CV_{int(index):03d}"


def parse_entity_index(value: str) -> int:
    match = re.search(r"(\d+)$", str(value))
    if not match:
        raise ValueError(f"Invalid entity identifier: {value}")
    return int(match.group(1))


@dataclass
class RawData:
    jobs: pd.DataFrame
    candidates: pd.DataFrame
    audit: dict


@dataclass
class SampledData:
    jobs: pd.DataFrame
    candidates: pd.DataFrame
    pairs: pd.DataFrame


def load_raw_data(data_dir: str | Path) -> RawData:
    paths = resolve_data_paths(data_dir)
    root = Path(data_dir).expanduser().resolve()
    input_files = build_input_manifest(root)
    jobs_raw = pd.read_csv(paths["jobs"])
    candidates_raw = pd.read_csv(paths["candidates"])
    missing_jobs = set(JOB_REQUIRED_COLUMNS) - set(jobs_raw.columns)
    missing_candidates = set(CANDIDATE_REQUIRED_COLUMNS) - set(candidates_raw.columns)
    if missing_jobs or missing_candidates:
        raise ValueError(
            f"Missing raw columns: jobs={sorted(missing_jobs)}, "
            f"candidates={sorted(missing_candidates)}"
        )

    duplicate_candidates = int(candidates_raw.duplicated().sum())
    jobs = jobs_raw.assign(_source_index=jobs_raw.index).copy()
    candidates = (
        candidates_raw.assign(_source_index=candidates_raw.index)
        .drop_duplicates(subset=[column for column in candidates_raw.columns], keep="first")
        .reset_index(drop=True)
    )
    if jobs["JobID"].duplicated().any() or jobs["URL Job"].duplicated().any():
        raise ValueError("Job identifiers or URLs are not unique")
    if candidates["UserID"].duplicated().any() or candidates["URL User"].duplicated().any():
        raise ValueError("Candidate identifiers or URLs remain duplicated after deduplication")

    audit = {
        "data_root": str(root),
        "input_files": input_files,
        "jobs_raw_rows": int(len(jobs_raw)),
        "jobs_clean_rows": int(len(jobs)),
        "jobs_columns": int(jobs_raw.shape[1]),
        "jobs_unique": int(jobs["JobID"].nunique()),
        "jobs_exact_duplicates": int(jobs_raw.duplicated().sum()),
        "jobs_missing_cells": int(jobs_raw.isna().sum().sum()),
        "jobs_required_columns": list(JOB_REQUIRED_COLUMNS),
        "candidates_raw_rows": int(len(candidates_raw)),
        "candidates_clean_rows": int(len(candidates)),
        "candidates_columns": int(candidates_raw.shape[1]),
        "candidates_unique": int(candidates["UserID"].nunique()),
        "candidates_exact_duplicates": duplicate_candidates,
        "candidates_missing_cells": int(candidates_raw.isna().sum().sum()),
        "candidates_required_columns": list(CANDIDATE_REQUIRED_COLUMNS),
    }
    return RawData(jobs=jobs, candidates=candidates, audit=audit)


def load_gold_with_identity_check(
    gold_path: str | Path,
    raw: RawData,
) -> pd.DataFrame:
    gold = pd.read_csv(gold_path)
    required = {"job_id", "cand_id", "job_title", "desired_job", "relevance"}
    missing = required - set(gold.columns)
    if missing:
        raise ValueError(f"Gold columns missing: {sorted(missing)}")
    job_indices = gold["job_id"].map(parse_entity_index)
    candidate_indices = gold["cand_id"].map(parse_entity_index)
    jobs_by_source = raw.jobs.set_index("_source_index")
    candidates_by_source = raw.candidates.set_index("_source_index")
    if not set(job_indices).issubset(jobs_by_source.index):
        raise ValueError("Gold contains an unknown raw job index")
    if not set(candidate_indices).issubset(candidates_by_source.index):
        raise ValueError("Gold contains an unknown or deduplicated candidate index")
    expected_jobs = job_indices.map(jobs_by_source["Job Title"]).fillna("").str.strip()
    expected_candidates = candidate_indices.map(candidates_by_source["Desired Job"]).fillna("").str.strip()
    if not expected_jobs.equals(gold["job_title"].fillna("").str.strip()):
        raise ValueError("Gold job IDs do not match raw job titles")
    if not expected_candidates.equals(gold["desired_job"].fillna("").str.strip()):
        raise ValueError("Gold candidate IDs do not match raw desired jobs")
    if not gold["relevance"].isin([0, 1, 2, 3]).all():
        raise ValueError("Gold relevance must be in {0,1,2,3}")
    if gold.duplicated(["job_id", "cand_id"]).any():
        raise ValueError("Gold contains duplicate CV--job pairs")
    return gold.reset_index(drop=True)


def make_gold_split_manifest(
    gold: pd.DataFrame,
    n_validation_jobs: int = 4,
    seed: int = 42,
) -> dict:
    jobs = sorted(gold["job_id"].unique())
    if len(jobs) < 2:
        raise ValueError("Gold requires at least two jobs for query-disjoint validation/test")
    if not 1 <= n_validation_jobs < len(jobs):
        raise ValueError(
            "n_validation_jobs must leave at least one Gold-test job"
        )
    shuffled_jobs = np.asarray(jobs, dtype=object)
    rng = np.random.RandomState(seed)
    rng.shuffle(shuffled_jobs)
    validation_jobs = sorted(shuffled_jobs[:n_validation_jobs].tolist())
    test_jobs = sorted(set(jobs) - set(validation_jobs))
    manifest = {
        "protocol": "query-disjoint-gold-label-blind-v3",
        "split_algorithm": "sorted-job-id-seeded-shuffle-v1",
        "seed": int(seed),
        "total_jobs": len(jobs),
        "validation_jobs": validation_jobs,
        "test_jobs": test_jobs,
        "validation_pairs": int(gold["job_id"].isin(validation_jobs).sum()),
        "test_pairs": int(gold["job_id"].isin(test_jobs).sum()),
    }
    if set(validation_jobs) & set(test_jobs):
        raise AssertionError("Gold validation/test jobs overlap")
    return manifest


def inter_annotator_agreement(
    annotation_path: str | Path | None,
    gold: pd.DataFrame,
) -> dict:
    if annotation_path is None:
        return {
            "status": "not_available",
            "reason": "No independent annotation file configured",
        }
    path = Path(annotation_path).expanduser().resolve()
    if not path.is_file():
        return {
            "status": "not_available",
            "reason": f"Independent annotation file not found: {path}",
        }

    annotations = pd.read_csv(path)
    required = {"job_id", "cand_id", "annotator_id", "relevance"}
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"Annotation columns missing: {sorted(missing)}")
    annotations = annotations[list(required)].copy()
    annotations["annotator_id"] = annotations["annotator_id"].astype(str).str.strip()
    if annotations["annotator_id"].eq("").any():
        raise ValueError("annotator_id must not be empty")
    if not annotations["relevance"].isin([0, 1, 2, 3]).all():
        raise ValueError("Annotation relevance must be in {0,1,2,3}")
    if annotations.duplicated(["job_id", "cand_id", "annotator_id"]).any():
        raise ValueError("Duplicate annotation for an annotator and CV--job pair")

    gold_pairs = set(map(tuple, gold[["job_id", "cand_id"]].itertuples(index=False, name=None)))
    annotation_pairs = set(map(tuple, annotations[["job_id", "cand_id"]].itertuples(index=False, name=None)))
    unknown = sorted(annotation_pairs - gold_pairs)
    if unknown:
        raise ValueError(f"Annotations contain pairs outside Gold: {unknown[:3]}")
    annotators = sorted(annotations["annotator_id"].unique())
    if len(annotators) != 2:
        raise ValueError("IAA requires exactly two independent annotators")

    paired = annotations.pivot(
        index=["job_id", "cand_id"], columns="annotator_id", values="relevance"
    ).dropna()
    if paired.empty:
        raise ValueError("Annotators have no overlapping CV--job pairs")
    left = paired[annotators[0]].astype(int)
    right = paired[annotators[1]].astype(int)
    return {
        "status": "available",
        "annotation_path": str(path),
        "annotators": annotators,
        "overlap_pairs": int(len(paired)),
        "gold_pairs": int(len(gold)),
        "coverage": float(len(paired) / len(gold)),
        "exact_agreement": float((left == right).mean()),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(left, right, labels=[0, 1, 2, 3], weights="quadratic")
        ),
    }


def sample_development_entities(
    raw: RawData,
    n_jobs: int,
    n_candidates: int,
    candidates_per_job: int,
    seed: int,
    excluded_job_ids: set[str],
    excluded_candidate_ids: set[str],
) -> SampledData:
    excluded_job_indices = {parse_entity_index(value) for value in excluded_job_ids}
    excluded_candidate_indices = {parse_entity_index(value) for value in excluded_candidate_ids}
    jobs = raw.jobs[~raw.jobs["_source_index"].isin(excluded_job_indices)]
    candidates = raw.candidates[~raw.candidates["_source_index"].isin(excluded_candidate_indices)]
    if n_jobs > len(jobs) or n_candidates > len(candidates):
        raise ValueError("Requested sample exceeds non-Gold entities")
    jobs = jobs.sample(n=n_jobs, random_state=seed).reset_index(drop=True)
    candidates = candidates.sample(n=n_candidates, random_state=seed).reset_index(drop=True)
    rng = np.random.RandomState(seed)
    rows = []
    for _, job in jobs.iterrows():
        selected = rng.choice(
            len(candidates),
            size=min(candidates_per_job, len(candidates)),
            replace=False,
        )
        for candidate_position in selected:
            candidate = candidates.iloc[int(candidate_position)]
            rows.append({
                "pair_id": len(rows),
                "job_id": format_job_id(job["_source_index"]),
                "cand_id": format_candidate_id(candidate["_source_index"]),
                "job_source_index": int(job["_source_index"]),
                "candidate_source_index": int(candidate["_source_index"]),
            })
    pairs = pd.DataFrame(rows)
    if pairs.duplicated(["job_id", "cand_id"]).any():
        raise AssertionError("Duplicate development pair")
    return SampledData(jobs=jobs, candidates=candidates, pairs=pairs)


def split_development_pairs(
    pairs: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    jobs = np.array(sorted(pairs["job_id"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(jobs)
    n_train = int(len(jobs) * train_ratio)
    n_validation = int(len(jobs) * validation_ratio)
    train_jobs = set(jobs[:n_train])
    validation_jobs = set(jobs[n_train:n_train + n_validation])
    test_jobs = set(jobs[n_train + n_validation:])
    if not train_jobs or not validation_jobs or not test_jobs:
        raise ValueError("Development split produced an empty partition")
    if train_jobs & validation_jobs or train_jobs & test_jobs or validation_jobs & test_jobs:
        raise AssertionError("Development jobs overlap")

    def select(values: set[str]) -> pd.DataFrame:
        return pairs[pairs["job_id"].isin(values)].copy().reset_index(drop=True)

    manifest = {
        "train_jobs": sorted(train_jobs),
        "validation_jobs": sorted(validation_jobs),
        "test_jobs": sorted(test_jobs),
    }
    return select(train_jobs), select(validation_jobs), select(test_jobs), manifest


def gold_pairs_to_raw_indices(gold: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "pair_id": gold.get("pair_id", pd.Series(range(len(gold)))),
        "job_id": gold["job_id"].to_numpy(),
        "cand_id": gold["cand_id"].to_numpy(),
        "job_source_index": gold["job_id"].map(parse_entity_index).to_numpy(int),
        "candidate_source_index": gold["cand_id"].map(parse_entity_index).to_numpy(int),
    })
