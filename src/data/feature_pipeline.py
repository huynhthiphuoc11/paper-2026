from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.loader import FEATURE_COLS, RealKaggleDatasetAdapter
from src.skill_normalization import extract_normalized_skills


def _extract_skills(text: str) -> set[str]:
    return extract_normalized_skills(text)


def _parse_experience(text: str) -> float:
    values = re.findall(r"\d+", str(text))
    return float(values[0]) if values else 1.0


def _check_location_overlap(candidate_location: str, job_location: str):
    if (
        not candidate_location
        or not job_location
        or candidate_location == "nan"
        or job_location == "nan"
    ):
        return np.nan
    cities = [
        "hà nội",
        "hồ chí minh",
        "tphcm",
        "đà nẵng",
        "cần thơ",
        "hải phòng",
        "bình dương",
    ]
    for city in cities:
        if city in candidate_location and city in job_location:
            return True
    return candidate_location in job_location or job_location in candidate_location


@dataclass
class SampledEntities:
    jobs: pd.DataFrame
    candidates: pd.DataFrame
    pairs: pd.DataFrame


def _with_source_index(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(_source_index=frame.index).reset_index(drop=True)


def load_gold_with_identity_check(data_dir: str, gold_path: str) -> pd.DataFrame:
    jobs = pd.read_csv(f"{data_dir}/JOB_DATA_FINAL.csv")
    candidates = pd.read_csv(f"{data_dir}/USER_DATA_FINAL.csv")
    gold = pd.read_csv(gold_path)
    if "cand_id" not in gold.columns and "user_id" in gold.columns:
        gold = gold.rename(columns={"user_id": "cand_id"})

    job_indices = gold["job_id"].map(RealKaggleDatasetAdapter.parse_id_index)
    candidate_indices = gold["cand_id"].map(RealKaggleDatasetAdapter.parse_id_index)
    if not job_indices.between(0, len(jobs) - 1).all():
        raise ValueError("Graded JOB raw index is outside JOB_DATA_FINAL.csv")
    if not candidate_indices.between(0, len(candidates) - 1).all():
        raise ValueError("Graded CV raw index is outside USER_DATA_FINAL.csv")

    expected_titles = job_indices.map(jobs["Job Title"])
    expected_desired = candidate_indices.map(candidates["Desired Job"])
    if not gold["job_title"].fillna("").str.strip().equals(
        expected_titles.fillna("").str.strip()
    ):
        raise ValueError("Graded JOB IDs do not match annotated job_title")
    if not gold["desired_job"].fillna("").str.strip().equals(
        expected_desired.fillna("").str.strip()
    ):
        raise ValueError("Graded CV IDs do not match annotated desired_job")
    return gold


def sample_raw_entities(
    data_dir: str,
    n_jobs: int,
    n_candidates: int,
    candidates_per_job: int,
    seed: int,
    excluded_job_ids: set[str] | None = None,
    excluded_candidate_ids: set[str] | None = None,
) -> SampledEntities:
    jobs_all = pd.read_csv(f"{data_dir}/JOB_DATA_FINAL.csv")
    candidates_all = pd.read_csv(f"{data_dir}/USER_DATA_FINAL.csv")
    jobs = _with_source_index(
        jobs_all.dropna(subset=["Job Title", "Job Requirements"])
    )
    candidates = _with_source_index(
        candidates_all.dropna(subset=["Desired Job", "Skills"])
    )

    excluded_raw_jobs = {
        RealKaggleDatasetAdapter.parse_id_index(value)
        for value in (excluded_job_ids or set())
    }
    excluded_raw_candidates = {
        RealKaggleDatasetAdapter.parse_id_index(value)
        for value in (excluded_candidate_ids or set())
    }
    jobs = jobs[~jobs["_source_index"].isin(excluded_raw_jobs)]
    candidates = candidates[
        ~candidates["_source_index"].isin(excluded_raw_candidates)
    ]
    jobs = jobs.sample(n=min(n_jobs, len(jobs)), random_state=seed).reset_index(drop=True)
    candidates = candidates.sample(
        n=min(n_candidates, len(candidates)), random_state=seed
    ).reset_index(drop=True)

    rng = np.random.RandomState(seed)
    rows = []
    for _, job in jobs.iterrows():
        selected = rng.choice(
            len(candidates),
            size=min(candidates_per_job, len(candidates)),
            replace=False,
        )
        job_id = f"JOB_{int(job['_source_index']):02d}"
        for candidate_position in selected:
            candidate = candidates.iloc[candidate_position]
            rows.append(
                {
                    "pair_id": len(rows),
                    "job_id": job_id,
                    "cand_id": f"CV_{int(candidate['_source_index']):03d}",
                    "job_source_index": int(job["_source_index"]),
                    "candidate_source_index": int(candidate["_source_index"]),
                }
            )
    pairs = pd.DataFrame(rows)
    if pairs.duplicated(["job_id", "cand_id"]).any():
        raise AssertionError("Duplicate sampled CV-JD pair")
    return SampledEntities(jobs=jobs, candidates=candidates, pairs=pairs)


def split_pairs_by_job(
    pairs: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    jobs = np.array(sorted(pairs["job_id"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(jobs)
    n_train = max(1, int(len(jobs) * train_ratio))
    n_val = max(1, int(len(jobs) * val_ratio))
    train_jobs = set(jobs[:n_train])
    val_jobs = set(jobs[n_train : n_train + n_val])
    test_jobs = set(jobs[n_train + n_val :])
    if not test_jobs:
        moved = sorted(train_jobs)[-1]
        train_jobs.remove(moved)
        test_jobs.add(moved)
    assert train_jobs.isdisjoint(val_jobs)
    assert train_jobs.isdisjoint(test_jobs)
    assert val_jobs.isdisjoint(test_jobs)

    def select(ids: set[str]) -> pd.DataFrame:
        return pairs[pairs["job_id"].isin(ids)].copy().reset_index(drop=True)

    metadata = {
        "train_jobs": sorted(train_jobs),
        "validation_jobs": sorted(val_jobs),
        "test_jobs": sorted(test_jobs),
    }
    return select(train_jobs), select(val_jobs), select(test_jobs), metadata


class TrainOnlyFeaturePipeline:
    def __init__(
        self,
        df_threshold: float | None = None,
        role_max_features: int = 2000,
        description_max_features: int = 5000,
    ):
        self.df_threshold = df_threshold
        self.role_vectorizer = TfidfVectorizer(max_features=role_max_features)
        self.desc_vectorizer = TfidfVectorizer(
            max_features=description_max_features
        )
        self.high_df_skills: set[str] = set()
        self.fit_job_ids: set[str] = set()
        self.fit_candidate_ids: set[str] = set()
        self.is_fitted = False

    @staticmethod
    def _job_id(row: pd.Series) -> str:
        return f"JOB_{int(row['_source_index']):02d}"

    @staticmethod
    def _candidate_id(row: pd.Series) -> str:
        return f"CV_{int(row['_source_index']):03d}"

    @staticmethod
    def _job_desc(row: pd.Series) -> str:
        return f"{row.get('Job Description', '')} {row.get('Job Requirements', '')}"

    @staticmethod
    def _candidate_profile(row: pd.Series) -> str:
        return f"{row.get('Target', '')} {row.get('Skills', '')} {row.get('Work Experience', '')}"

    def fit(
        self,
        train_pairs: pd.DataFrame,
        jobs: pd.DataFrame,
        candidates: pd.DataFrame,
    ) -> "TrainOnlyFeaturePipeline":
        job_indices = set(train_pairs["job_source_index"].astype(int))
        candidate_indices = set(train_pairs["candidate_source_index"].astype(int))
        train_jobs = jobs[jobs["_source_index"].isin(job_indices)]
        train_candidates = candidates[candidates["_source_index"].isin(candidate_indices)]
        self.fit_job_ids = {self._job_id(row) for _, row in train_jobs.iterrows()}
        self.fit_candidate_ids = {
            self._candidate_id(row) for _, row in train_candidates.iterrows()
        }
        self.role_vectorizer.fit(
            train_jobs["Job Title"].fillna("").tolist()
            + train_candidates["Desired Job"].fillna("").tolist()
        )
        self.desc_vectorizer.fit(
            [self._job_desc(row) for _, row in train_jobs.iterrows()]
            + [self._candidate_profile(row) for _, row in train_candidates.iterrows()]
        )

        if self.df_threshold is not None:
            documents = [
                _extract_skills(str(value))
                for value in train_jobs["Job Requirements"].fillna("")
            ] + [
                _extract_skills(str(value))
                for value in train_candidates["Skills"].fillna("")
            ]
            counts: dict[str, int] = {}
            for document in documents:
                for skill in document:
                    counts[skill] = counts.get(skill, 0) + 1
            denominator = max(1, len(documents))
            self.high_df_skills = {
                skill
                for skill, count in counts.items()
                if count / denominator > self.df_threshold
            }
        self.is_fitted = True
        return self

    def _feature_row(self, job: pd.Series, candidate: pd.Series) -> dict:
        job_skills = _extract_skills(str(job.get("Job Requirements", "")))
        candidate_skills = _extract_skills(str(candidate.get("Skills", "")))
        job_skills -= self.high_df_skills
        candidate_skills -= self.high_df_skills
        required_years = _parse_experience(
            str(job.get("Years of Experience", "1"))
        )
        candidate_years = _parse_experience(
            str(candidate.get("Work Experience", "1"))
        )
        overlap = _check_location_overlap(
            str(candidate.get("Workplace Desired", "")).lower(),
            str(job.get("Job Address", "Hanoi")).lower(),
        )
        loc_match = np.nan if pd.isna(overlap) else float(bool(overlap))
        union = max(1, len(job_skills | candidate_skills))
        skill_iou = len(job_skills & candidate_skills) / union
        experience_gap = abs(candidate_years - required_years)
        exp_score = float(np.exp(-experience_gap / max(1.0, required_years)))
        role_match = float(
            np.clip(
                cosine_similarity(
                    self.role_vectorizer.transform([str(candidate.get("Desired Job", ""))]),
                    self.role_vectorizer.transform([str(job.get("Job Title", ""))]),
                )[0, 0]
                * 1.5,
                0.0,
                1.0,
            )
        )
        desc_sem_sim = float(
            np.clip(
                cosine_similarity(
                    self.desc_vectorizer.transform([self._candidate_profile(candidate)]),
                    self.desc_vectorizer.transform([self._job_desc(job)]),
                )[0, 0]
                * 1.3,
                0.0,
                1.0,
            )
        )
        heuristic_score = (
            0.30 * loc_match
            + 0.25 * skill_iou
            + 0.20 * exp_score
            + 0.15 * role_match
            + 0.10 * desc_sem_sim
        )
        return {
            "job_title": str(job.get("Job Title", "")),
            "loc_match": loc_match,
            "skill_iou": float(skill_iou),
            "exp_score": exp_score,
            "role_match": role_match,
            "desc_sem_sim": desc_sem_sim,
            "heuristic_score": float(heuristic_score),
            "heuristic_label": int(heuristic_score >= 0.45),
            "experience_gap": float(experience_gap),
            "job_required_years": float(required_years),
            "cv_years": float(candidate_years),
            "required_skill_match_ratio": float(
                len(job_skills & candidate_skills) / max(1, len(job_skills))
            ),
            "missing_required_skill_ratio": float(
                len(job_skills - candidate_skills) / max(1, len(job_skills))
            ),
            "job_skills": " ".join(sorted(job_skills)),
            "user_skills": " ".join(sorted(candidate_skills)),
        }

    def transform(
        self,
        pairs: pd.DataFrame,
        jobs: pd.DataFrame,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Fit TrainOnlyFeaturePipeline on train before transform")
        job_lookup = jobs.set_index("_source_index")
        candidate_lookup = candidates.set_index("_source_index")
        rows = []
        for pair in pairs.itertuples(index=False):
            job = job_lookup.loc[int(pair.job_source_index)]
            candidate = candidate_lookup.loc[int(pair.candidate_source_index)]
            rows.append({**pair._asdict(), **self._feature_row(job, candidate)})
        frame = pd.DataFrame(rows)
        if not np.isfinite(frame[FEATURE_COLS].to_numpy(float)).all():
            raise ValueError("Non-finite model feature after transform")
        return frame

    def transform_gold(
        self,
        gold: pd.DataFrame,
        data_dir: str,
    ) -> pd.DataFrame:
        jobs = _with_source_index(pd.read_csv(f"{data_dir}/JOB_DATA_FINAL.csv"))
        candidates = _with_source_index(pd.read_csv(f"{data_dir}/USER_DATA_FINAL.csv"))
        pairs = pd.DataFrame(
            {
                "pair_id": gold.get("pair_id", pd.Series(range(len(gold)))),
                "job_id": gold["job_id"],
                "cand_id": gold["cand_id"],
                "job_source_index": gold["job_id"].map(
                    RealKaggleDatasetAdapter.parse_id_index
                ),
                "candidate_source_index": gold["cand_id"].map(
                    RealKaggleDatasetAdapter.parse_id_index
                ),
            }
        )
        features = self.transform(pairs, jobs, candidates)
        overlap = [
            column
            for column in features.columns
            if column in gold.columns
            and column not in {"pair_id", "job_id", "cand_id"}
        ]
        base = gold.drop(columns=overlap)
        merged = base.merge(
            features, on=["pair_id", "job_id", "cand_id"], how="inner"
        )
        if len(merged) != len(gold):
            raise ValueError("Graded feature transform did not preserve complete coverage")
        return merged

    def manifest(self) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Feature pipeline is not fitted")
        role_terms = "\n".join(sorted(self.role_vectorizer.vocabulary_))
        desc_terms = "\n".join(sorted(self.desc_vectorizer.vocabulary_))
        return {
            "fit_job_ids": sorted(self.fit_job_ids),
            "fit_candidate_ids": sorted(self.fit_candidate_ids),
            "role_vocabulary_size": len(self.role_vectorizer.vocabulary_),
            "desc_vocabulary_size": len(self.desc_vectorizer.vocabulary_),
            "role_vocabulary_sha256": hashlib.sha256(role_terms.encode()).hexdigest(),
            "desc_vocabulary_sha256": hashlib.sha256(desc_terms.encode()).hexdigest(),
            "high_df_skills": sorted(self.high_df_skills),
            "feature_cols": list(FEATURE_COLS),
        }
