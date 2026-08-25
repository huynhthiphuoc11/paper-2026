from __future__ import annotations

import hashlib
import importlib.metadata
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.experiments import FEATURE_COLUMNS
from src.experiments.data import RawData, gold_pairs_to_raw_indices
from src.experiments.utils import normalize_text


# Predefined, manually curated skill normalization — fixed before experimentation,
# applied identically to every partition (not fit on train). Only TF-IDF
# vocabularies are train-fitted.
_SKILL_ALIASES = {
    "python3": "python", "python 3": "python", "py": "python",
    "machine-learning": "machine learning", "ml": "machine learning",
    "deep-learning": "deep learning", "dl": "deep learning",
    "powerbi": "power bi", "power-bi": "power bi",
    "nodejs": "node.js", "postgresql": "postgres", "sql server": "sql",
    "ms office": "microsoft office", "tin học văn phòng": "microsoft office",
}
_KNOWN_SKILLS = sorted(set(_SKILL_ALIASES.values()) | {
    "python", "java", "javascript", "typescript", "c++", "c#", "php", "sql",
    "excel", "microsoft office", "power bi", "tableau", "machine learning",
    "deep learning", "data analysis", "data science", "computer vision",
    "natural language processing", "project management", "sales", "marketing",
    "seo", "accounting", "logistics", "human resources", "customer service",
    "communication", "teamwork", "leadership", "english", "chinese", "japanese",
}, key=len, reverse=True)


def normalize_location(value) -> str:
    text = normalize_text(value)
    replacements = {
        "tp.hcm": "hồ chí minh", "tphcm": "hồ chí minh", "tp hcm": "hồ chí minh",
        "hcm": "hồ chí minh", "hn": "hà nội",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def normalize_skill(value: str) -> str:
    skill = normalize_text(value)
    skill = re.sub(r"^[\W_]+|[\W_]+$", "", skill)
    skill = re.sub(r"\s+", " ", skill)
    return _SKILL_ALIASES.get(skill, skill)


def extract_skill_set(value) -> set[str]:
    text = normalize_text(value)
    if not text:
        return set()
    found = {
        phrase for phrase in _KNOWN_SKILLS
        if re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text)
    }
    for item in re.split(r"[,;|/\n•]+", text):
        candidate = normalize_skill(item)
        if (
            2 <= len(candidate) <= 60
            and re.search(r"[\wÀ-ỹ]", candidate)
            and len(candidate.split()) <= 7
            and candidate not in {"and", "with", "year", "years", "kỹ năng"}
        ):
            found.add(candidate)
    return found


def _parse_experience_numbers(text: str) -> tuple[float, float]:
    numbers = [
        float(number)
        for number in re.findall(r"\d+(?:[.,]\d+)?", text.replace(",", "."))
    ]
    if "trên" in text or "hơn" in text or ">" in text:
        lower = numbers[0] if numbers else 10.0
        return lower, float("inf")
    if len(numbers) >= 2:
        return min(numbers[0], numbers[1]), max(numbers[0], numbers[1])
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return 0.0, float("inf")


def parse_required_experience(value) -> tuple[float, float] | None:
    text = normalize_text(value)
    if not text:
        return None
    if "không yêu cầu" in text:
        return 0.0, float("inf")
    return _parse_experience_numbers(text)


def parse_candidate_experience(value) -> tuple[float, float] | None:
    text = normalize_text(value)
    if not text:
        return None
    if "chưa có" in text:
        return 0.0, 0.0
    return _parse_experience_numbers(text)


def interval_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    left_low, left_high = left
    right_low, right_high = right
    if left_high >= right_low and right_high >= left_low:
        return 0.0
    if left_high < right_low:
        return right_low - left_high
    return left_low - right_high


def experience_score(job_value, candidate_value) -> float:
    required = parse_required_experience(job_value)
    candidate = parse_candidate_experience(candidate_value)
    if required is None or candidate is None:
        return np.nan
    distance = interval_distance(required, candidate)
    return float(np.exp(-distance / max(1.0, required[0])))


def location_match(candidate_value, job_value) -> float:
    candidate = normalize_location(candidate_value)
    job = normalize_location(job_value)
    if not candidate or not job:
        return 0.0
    cities = [
        "hà nội", "hồ chí minh", "đà nẵng", "cần thơ", "hải phòng",
        "bình dương", "đồng nai", "bắc giang",
    ]
    return float(any(city in candidate and city in job for city in cities) or candidate in job or job in candidate)


@dataclass
class FeatureState:
    role_vectorizer: TfidfVectorizer
    description_vectorizer: TfidfVectorizer
    fit_job_ids: set[str]
    fit_candidate_ids: set[str]


class ThreeSignalFeaturePipeline:
    def __init__(
        self,
        semantic_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        semantic_batch_size: int = 64,
        semantic_device: str | None = None,
        semantic_max_features: int = 6000,
        role_max_features: int = 2000,
        semantic_encoder=None,
    ):
        self.semantic_model_name = semantic_model_name
        self.semantic_batch_size = int(semantic_batch_size)
        self.semantic_device = semantic_device
        self.semantic_encoder = semantic_encoder
        self.role_vectorizer = TfidfVectorizer(max_features=role_max_features, ngram_range=(1, 2))
        self.description_vectorizer = TfidfVectorizer(
            max_features=semantic_max_features, ngram_range=(1, 2)
        )
        self.fit_job_ids: set[str] = set()
        self.fit_candidate_ids: set[str] = set()
        self.embedding_dimension: int | None = None
        self.is_fitted = False

    def _get_semantic_encoder(self):
        if self.semantic_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise ImportError(
                    "Install sentence-transformers to compute the multilingual semantic signal"
                ) from error
            self.semantic_encoder = SentenceTransformer(
                self.semantic_model_name,
                device=self.semantic_device,
            )
        return self.semantic_encoder

    def _encode_semantic(self, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.empty((0, 0), dtype=np.float32)
        embeddings = np.asarray(
            self._get_semantic_encoder().encode(
                documents,
                batch_size=self.semantic_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )
        if embeddings.ndim != 2 or len(embeddings) != len(documents):
            raise ValueError("Semantic encoder returned an invalid embedding matrix")
        if not np.isfinite(embeddings).all():
            raise ValueError("Semantic encoder returned non-finite values")
        self.embedding_dimension = int(embeddings.shape[1])
        return embeddings

    @staticmethod
    def _job_semantic(row: pd.Series) -> str:
        return normalize_text(" ".join([
            str(row.get("Job Title", "") or ""),
            str(row.get("Job Description", "") or ""),
            str(row.get("Job Requirements", "") or ""),
        ]))

    @staticmethod
    def _candidate_semantic(row: pd.Series) -> str:
        return normalize_text(" ".join([
            str(row.get("Desired Job", "") or ""),
            str(row.get("Target", "") or ""),
            str(row.get("Skills", "") or ""),
        ]))

    @staticmethod
    def _job_description(row: pd.Series) -> str:
        return normalize_text(row.get("Job Description", ""))

    @staticmethod
    def _candidate_description(row: pd.Series) -> str:
        return normalize_text(row.get("Target", ""))

    def fit(self, train_pairs: pd.DataFrame, jobs: pd.DataFrame, candidates: pd.DataFrame):
        job_indices = set(train_pairs["job_source_index"].astype(int))
        candidate_indices = set(train_pairs["candidate_source_index"].astype(int))
        train_jobs = jobs[jobs["_source_index"].isin(job_indices)]
        train_candidates = candidates[candidates["_source_index"].isin(candidate_indices)]
        self.fit_job_ids = {f"JOB_{int(value)}" for value in train_jobs["_source_index"]}
        self.fit_candidate_ids = {f"CV_{int(value):03d}" for value in train_candidates["_source_index"]}
        role_documents = train_jobs["Job Title"].map(normalize_text).tolist()
        role_documents += train_candidates["Desired Job"].map(normalize_text).tolist()
        description_documents = [
            self._job_description(row) for _, row in train_jobs.iterrows()
        ]
        description_documents += [
            self._candidate_description(row) for _, row in train_candidates.iterrows()
        ]
        self.role_vectorizer.fit(role_documents)
        self.description_vectorizer.fit(description_documents)
        encoder = self._get_semantic_encoder()
        dimension = getattr(encoder, "get_sentence_embedding_dimension", lambda: None)()
        if dimension is None:
            probe = self._encode_semantic(["dimension probe"])
            dimension = probe.shape[1]
        self.embedding_dimension = int(dimension)
        self.is_fitted = True
        return self

    def _feature_row(self, job: pd.Series, candidate: pd.Series) -> dict:
        job_text = self._job_semantic(job)
        candidate_text = self._candidate_semantic(candidate)
        semantic_available = bool(job_text and candidate_text)
        if semantic_available:
            embeddings = self._encode_semantic([job_text, candidate_text])
            s_sem = float(np.dot(embeddings[0], embeddings[1]))
        else:
            s_sem = np.nan

        job_skills = extract_skill_set(job.get("Job Requirements", ""))
        candidate_skills = extract_skill_set(candidate.get("Skills", ""))
        skill_available = bool(job_skills) and bool(candidate_skills)
        union = job_skills | candidate_skills
        s_skill = (
            float(len(job_skills & candidate_skills) / len(union))
            if skill_available else np.nan
        )

        required_experience = parse_required_experience(
            job.get("Years of Experience", "")
        )
        candidate_experience = parse_candidate_experience(
            candidate.get("Work Experience", "")
        )
        experience_available = (
            required_experience is not None and candidate_experience is not None
        )
        s_exp = (
            experience_score(
                job.get("Years of Experience", ""),
                candidate.get("Work Experience", ""),
            )
            if experience_available else np.nan
        )

        role_score = float(cosine_similarity(
            self.role_vectorizer.transform([normalize_text(job.get("Job Title", ""))]),
            self.role_vectorizer.transform([normalize_text(candidate.get("Desired Job", ""))]),
        )[0, 0])
        description_score = float(cosine_similarity(
            self.description_vectorizer.transform([self._job_description(job)]),
            self.description_vectorizer.transform([self._candidate_description(candidate)]),
        )[0, 0])
        loc_score = location_match(candidate.get("Workplace Desired", ""), job.get("Job Address", ""))
        baseline_sem = 0.0 if not np.isfinite(s_sem) else s_sem
        baseline_skill = 0.0 if not np.isfinite(s_skill) else s_skill
        baseline_exp = 0.0 if not np.isfinite(s_exp) else s_exp
        heuristic_score = (
            0.30 * loc_score + 0.25 * baseline_skill + 0.20 * baseline_exp
            + 0.15 * role_score + 0.10 * description_score
        )
        return {
            "s_sem": s_sem,
            "s_skill": s_skill,
            "s_exp": s_exp,
            "sem_available": semantic_available,
            "skill_available": skill_available,
            "exp_available": experience_available,
            "baseline_location": loc_score,
            "baseline_skill": baseline_skill,
            "baseline_experience": baseline_exp,
            "baseline_role": role_score,
            "baseline_description": description_score,
            "heuristic_score": float(heuristic_score),
        }

    def transform(self, pairs: pd.DataFrame, jobs: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
        """Transform pairs with one vectorizer call per entity type, not per pair."""
        if not self.is_fitted:
            raise RuntimeError("Feature pipeline must be fitted on train")
        if pairs.empty:
            return pairs.copy()

        job_indices = pd.Index(pd.unique(pairs["job_source_index"].astype(int)))
        candidate_indices = pd.Index(pd.unique(pairs["candidate_source_index"].astype(int)))
        job_entities = jobs.set_index("_source_index").loc[job_indices]
        candidate_entities = candidates.set_index("_source_index").loc[candidate_indices]

        job_semantic_text = [self._job_semantic(row) for _, row in job_entities.iterrows()]
        candidate_semantic_text = [self._candidate_semantic(row) for _, row in candidate_entities.iterrows()]
        job_semantic_matrix = self._encode_semantic(job_semantic_text)
        candidate_semantic_matrix = self._encode_semantic(candidate_semantic_text)

        job_role_text = job_entities["Job Title"].map(normalize_text).tolist()
        candidate_role_text = candidate_entities["Desired Job"].map(normalize_text).tolist()
        job_role_matrix = self.role_vectorizer.transform(job_role_text)
        candidate_role_matrix = self.role_vectorizer.transform(candidate_role_text)

        job_description_text = [
            self._job_description(row) for _, row in job_entities.iterrows()
        ]
        candidate_description_text = [
            self._candidate_description(row) for _, row in candidate_entities.iterrows()
        ]
        job_description_matrix = self.description_vectorizer.transform(
            job_description_text
        )
        candidate_description_matrix = self.description_vectorizer.transform(
            candidate_description_text
        )

        job_positions = pairs["job_source_index"].astype(int).map(
            {value: position for position, value in enumerate(job_indices)}
        ).to_numpy(int)
        candidate_positions = pairs["candidate_source_index"].astype(int).map(
            {value: position for position, value in enumerate(candidate_indices)}
        ).to_numpy(int)

        semantic_available_by_job = np.asarray([bool(text) for text in job_semantic_text])
        semantic_available_by_candidate = np.asarray([bool(text) for text in candidate_semantic_text])
        semantic_available = (
            semantic_available_by_job[job_positions]
            & semantic_available_by_candidate[candidate_positions]
        )
        semantic_scores = np.einsum(
            "ij,ij->i",
            job_semantic_matrix[job_positions],
            candidate_semantic_matrix[candidate_positions],
        )
        semantic_scores[~semantic_available] = np.nan

        role_scores = np.asarray(
            job_role_matrix[job_positions]
            .multiply(candidate_role_matrix[candidate_positions])
            .sum(axis=1)
        ).ravel()
        description_scores = np.asarray(
            job_description_matrix[job_positions]
            .multiply(candidate_description_matrix[candidate_positions])
            .sum(axis=1)
        ).ravel()

        job_skills = [extract_skill_set(value) for value in job_entities["Job Requirements"]]
        candidate_skills = [extract_skill_set(value) for value in candidate_entities["Skills"]]
        job_experience = [normalize_text(value) for value in job_entities["Years of Experience"]]
        candidate_experience = [normalize_text(value) for value in candidate_entities["Work Experience"]]
        job_locations = [value for value in job_entities["Job Address"]]
        candidate_locations = [value for value in candidate_entities["Workplace Desired"]]

        skill_scores = np.empty(len(pairs), dtype=float)
        experience_scores = np.empty(len(pairs), dtype=float)
        location_scores = np.empty(len(pairs), dtype=float)
        skill_available = np.empty(len(pairs), dtype=bool)
        experience_available = np.empty(len(pairs), dtype=bool)
        for row_index, (job_position, candidate_position) in enumerate(
            zip(job_positions, candidate_positions)
        ):
            left_skills = job_skills[job_position]
            right_skills = candidate_skills[candidate_position]
            union = left_skills | right_skills
            skill_available[row_index] = bool(left_skills) and bool(right_skills)
            skill_scores[row_index] = (
                len(left_skills & right_skills) / len(union)
                if skill_available[row_index] else np.nan
            )
            job_exp = job_experience[job_position]
            candidate_exp = candidate_experience[candidate_position]
            experience_available[row_index] = (
                parse_required_experience(job_exp) is not None
                and parse_candidate_experience(candidate_exp) is not None
            )
            experience_scores[row_index] = (
                experience_score(job_exp, candidate_exp)
                if experience_available[row_index] else np.nan
            )
            location_scores[row_index] = location_match(
                candidate_locations[candidate_position], job_locations[job_position]
            )

        output = pairs.copy().reset_index(drop=True)
        output["s_sem"] = semantic_scores
        output["s_skill"] = skill_scores
        output["s_exp"] = experience_scores
        output["sem_available"] = semantic_available
        output["skill_available"] = skill_available
        output["exp_available"] = experience_available
        output["baseline_location"] = location_scores
        output["baseline_skill"] = np.nan_to_num(skill_scores, nan=0.0)
        output["baseline_experience"] = np.nan_to_num(experience_scores, nan=0.0)
        output["baseline_role"] = role_scores
        output["baseline_description"] = description_scores
        output["heuristic_score"] = (
            0.30 * output["baseline_location"]
            + 0.25 * output["baseline_skill"]
            + 0.20 * output["baseline_experience"]
            + 0.15 * output["baseline_role"]
            + 0.10 * output["baseline_description"]
        )
        return output

    def transform_gold(self, gold: pd.DataFrame, raw: RawData) -> pd.DataFrame:
        pairs = gold_pairs_to_raw_indices(gold)
        features = self.transform(pairs, raw.jobs, raw.candidates)
        metadata = gold.drop(columns=[column for column in features.columns if column in gold.columns and column not in {"pair_id", "job_id", "cand_id"}])
        merged = metadata.merge(features, on=["pair_id", "job_id", "cand_id"], how="inner")
        if len(merged) != len(gold):
            raise ValueError("Gold feature transform lost rows")
        return merged

    def impute_for_models(self, frame: pd.DataFrame, fill_values: dict[str, float] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
        output = frame.copy()
        if fill_values is None:
            fill_values = {
                column: float(output[column].median()) if output[column].notna().any() else 0.0
                for column in FEATURE_COLUMNS
            }
        output[FEATURE_COLUMNS] = output[FEATURE_COLUMNS].fillna(fill_values)
        if not np.isfinite(output[FEATURE_COLUMNS].to_numpy(float)).all():
            raise ValueError("Non-finite model feature")
        return output, dict(fill_values)

    @staticmethod
    def _sentence_transformers_version() -> str | None:
        try:
            return importlib.metadata.version("sentence-transformers")
        except importlib.metadata.PackageNotFoundError:
            return None

    def manifest(self) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Feature pipeline is not fitted")
        role_terms = "\n".join(sorted(self.role_vectorizer.vocabulary_))
        description_terms = "\n".join(
            sorted(self.description_vectorizer.vocabulary_)
        )
        return {
            "feature_columns": list(FEATURE_COLUMNS),
            "baseline_components": [
                "baseline_location", "baseline_skill", "baseline_experience",
                "baseline_role", "baseline_description",
            ],
            "signal_definition_version": "multilingual-sentence-embedding-v2",
            "semantic_encoder": {
                "model_name": self.semantic_model_name,
                "batch_size": self.semantic_batch_size,
                "device": self.semantic_device,
                "embedding_dimension": self.embedding_dimension,
                "sentence_transformers_version": self._sentence_transformers_version(),
            },
            "signal_field_mapping": {
                "s_sem": [
                    "Job Title + Job Description + Job Requirements",
                    "Desired Job + Target + Skills",
                ],
                "s_skill": ["Job Requirements", "Skills"],
                "s_exp": ["Years of Experience", "Work Experience"],
            },
            "baseline_definition_version": "independent-five-component-v2",
            "baseline_field_mapping": {
                "location": ["Job Address", "Workplace Desired"],
                "skill": ["Job Requirements", "Skills"],
                "experience": ["Years of Experience", "Work Experience"],
                "role": ["Job Title", "Desired Job"],
                "description": ["Job Description", "Target"],
            },
            "fit_job_ids": sorted(self.fit_job_ids),
            "fit_candidate_ids": sorted(self.fit_candidate_ids),
            "role_lexical_vocabulary_size": len(self.role_vectorizer.vocabulary_),
            "description_lexical_vocabulary_size": len(
                self.description_vectorizer.vocabulary_
            ),
            "role_lexical_vocabulary_sha256": hashlib.sha256(role_terms.encode()).hexdigest(),
            "description_lexical_vocabulary_sha256": hashlib.sha256(
                description_terms.encode()
            ).hexdigest(),
        }
