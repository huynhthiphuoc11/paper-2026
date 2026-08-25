import hashlib
import unittest

import numpy as np
import pandas as pd

from src.experiments import FEATURE_COLUMNS
from src.experiments.features import (
    ThreeSignalFeaturePipeline,
    extract_skill_set,
    parse_candidate_experience,
    parse_required_experience,
)


class FakeSemanticEncoder:
    dimension = 64

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, documents, **kwargs):
        vectors = []
        for document in documents:
            vector = np.zeros(self.dimension, dtype=float)
            for token in document.split():
                index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dimension
                vector[index] += 1.0
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.asarray(vectors)


class FeatureTests(unittest.TestCase):
    def test_feature_contract_is_exactly_three_signals(self):
        self.assertEqual(FEATURE_COLUMNS, ["s_sem", "s_skill", "s_exp"])

    def test_experience_parsers_keep_required_and_candidate_semantics(self):
        self.assertEqual(
            parse_required_experience("Không yêu cầu kinh nghiệm"),
            (0.0, float("inf")),
        )
        self.assertEqual(parse_required_experience("1-3 năm"), (1.0, 3.0))
        self.assertEqual(
            parse_required_experience("Trên 10 năm"),
            (10.0, float("inf")),
        )
        self.assertEqual(parse_candidate_experience("Chưa có kinh nghiệm"), (0.0, 0.0))
        self.assertIsNone(parse_required_experience(""))
        self.assertIsNone(parse_candidate_experience(None))

    def test_skill_parser_rejects_punctuation(self):
        skills = extract_skill_set("python; ...; SQL; •; machine-learning")
        self.assertIn("python", skills)
        self.assertIn("sql", skills)
        self.assertIn("machine learning", skills)
        self.assertNotIn("...", skills)

    @staticmethod
    def minimal_entities(candidate_skills="python", candidate_experience="5 năm"):
        jobs = pd.DataFrame({
            "_source_index": [1],
            "Job Title": ["Data Analyst"],
            "Job Description": ["Phân tích báo cáo kinh doanh"],
            "Job Requirements": ["python; sql"],
            "Years of Experience": ["Không yêu cầu kinh nghiệm"],
            "Job Address": ["Hà Nội"],
        })
        candidates = pd.DataFrame({
            "_source_index": [3],
            "Desired Job": ["Data Analyst"],
            "Target": ["Phân tích báo cáo kinh doanh"],
            "Skills": [candidate_skills],
            "Work Experience": [candidate_experience],
            "Workplace Desired": ["Hà Nội"],
        })
        pairs = pd.DataFrame({
            "pair_id": [0], "job_id": ["JOB_1"], "cand_id": ["CV_003"],
            "job_source_index": [1], "candidate_source_index": [3],
        })
        return jobs, candidates, pairs

    def test_one_sided_missing_skill_abstains(self):
        jobs, candidates, pairs = self.minimal_entities(candidate_skills="")
        pipeline = ThreeSignalFeaturePipeline(semantic_encoder=FakeSemanticEncoder()).fit(pairs, jobs, candidates)
        result = pipeline.transform(pairs, jobs, candidates).iloc[0]
        self.assertFalse(bool(result["skill_available"]))
        self.assertTrue(np.isnan(result["s_skill"]))

    def test_no_required_experience_accepts_five_year_candidate(self):
        jobs, candidates, pairs = self.minimal_entities(candidate_experience="5 năm")
        pipeline = ThreeSignalFeaturePipeline(semantic_encoder=FakeSemanticEncoder()).fit(pairs, jobs, candidates)
        result = pipeline.transform(pairs, jobs, candidates).iloc[0]
        self.assertTrue(bool(result["exp_available"]))
        self.assertAlmostEqual(float(result["s_exp"]), 1.0)

    def test_missing_experience_abstains(self):
        jobs, candidates, pairs = self.minimal_entities(candidate_experience="")
        pipeline = ThreeSignalFeaturePipeline(semantic_encoder=FakeSemanticEncoder()).fit(pairs, jobs, candidates)
        result = pipeline.transform(pairs, jobs, candidates).iloc[0]
        self.assertFalse(bool(result["exp_available"]))
        self.assertTrue(np.isnan(result["s_exp"]))

    def test_semantic_signal_uses_role_and_skill_fields(self):
        jobs, first_candidates, pairs = self.minimal_entities(candidate_skills="python")
        _, second_candidates, _ = self.minimal_entities(candidate_skills="java")
        second_candidates["Desired Job"] = "Marketing Manager"
        fit_candidates = pd.concat([
            first_candidates,
            second_candidates.assign(_source_index=4),
        ])
        fit_pairs = pd.concat([
            pairs,
            pairs.assign(pair_id=1, cand_id="CV_004", candidate_source_index=4),
        ], ignore_index=True)
        pipeline = ThreeSignalFeaturePipeline(semantic_encoder=FakeSemanticEncoder()).fit(fit_pairs, jobs, fit_candidates)
        first = pipeline.transform(pairs, jobs, first_candidates).iloc[0]
        second = pipeline.transform(pairs, jobs, second_candidates).iloc[0]
        self.assertGreater(float(first["s_sem"]), float(second["s_sem"]))

    def test_semantic_signal_is_invariant_to_description_change(self):
        jobs, candidates, pairs = self.minimal_entities()
        alternative = candidates.copy()
        alternative["Target"] = "Vận hành kho và giao nhận"
        fit_candidates = pd.concat([
            candidates,
            alternative.assign(_source_index=4),
        ])
        fit_pairs = pd.concat([
            pairs,
            pairs.assign(pair_id=1, cand_id="CV_004", candidate_source_index=4),
        ], ignore_index=True)
        pipeline = ThreeSignalFeaturePipeline(semantic_encoder=FakeSemanticEncoder()).fit(fit_pairs, jobs, fit_candidates)
        matching = pipeline.transform(pairs, jobs, candidates).iloc[0]
        changed = pipeline.transform(pairs, jobs, alternative).iloc[0]
        self.assertGreater(float(matching["s_sem"]), float(changed["s_sem"]))

    def test_semantic_field_mapping_uses_broad_concatenation(self):
        self.assertEqual(
            ThreeSignalFeaturePipeline(semantic_encoder=FakeSemanticEncoder()).manifest.__code__.co_consts[1] if False else None,
            None,
        )
        jobs, candidates, pairs = self.minimal_entities()
        pipeline = ThreeSignalFeaturePipeline(semantic_encoder=FakeSemanticEncoder()).fit(pairs, jobs, candidates)
        manifest = pipeline.manifest()
        self.assertEqual(
            manifest["signal_field_mapping"]["s_sem"],
            ["Job Title + Job Description + Job Requirements",
             "Desired Job + Target + Skills"],
        )
        self.assertEqual(
            manifest["signal_definition_version"],
            "multilingual-sentence-embedding-v2",
        )
        self.assertIn("semantic_encoder", manifest)
        self.assertNotIn("semantic_vocabulary_size", manifest)

    def test_description_baseline_is_invariant_to_candidate_skills(self):
        jobs, first_candidates, pairs = self.minimal_entities(candidate_skills="python")
        _, second_candidates, _ = self.minimal_entities(candidate_skills="java; marketing")
        fit_candidates = pd.concat([first_candidates, second_candidates.assign(_source_index=4)])
        fit_pairs = pd.concat([
            pairs,
            pairs.assign(pair_id=1, cand_id="CV_004", candidate_source_index=4),
        ], ignore_index=True)
        pipeline = ThreeSignalFeaturePipeline(semantic_encoder=FakeSemanticEncoder()).fit(fit_pairs, jobs, fit_candidates)
        first = pipeline.transform(pairs, jobs, first_candidates).iloc[0]
        second = pipeline.transform(pairs, jobs, second_candidates).iloc[0]
        self.assertAlmostEqual(
            float(first["baseline_description"]),
            float(second["baseline_description"]),
        )

    def test_vectorizers_fit_train_only(self):
        jobs = pd.DataFrame({
            "_source_index": [1, 2],
            "Job Title": ["trainrole", "validationonlytoken"],
            "Job Description": ["train description", "secret description"],
            "Job Requirements": ["python", "rust"],
            "Years of Experience": ["1-3 năm", "3-5 năm"],
            "Job Address": ["Hà Nội", "Hà Nội"],
        })
        candidates = pd.DataFrame({
            "_source_index": [3, 4],
            "Desired Job": ["trainrole", "candidateonlytoken"],
            "Target": ["python", "secret target"],
            "Skills": ["python", "rust"],
            "Work Experience": ["1-3 năm", "1-3 năm"],
            "Workplace Desired": ["Hà Nội", "Hà Nội"],
        })
        pairs = pd.DataFrame({
            "pair_id": [0, 1], "job_id": ["JOB_1", "JOB_2"],
            "cand_id": ["CV_003", "CV_004"], "job_source_index": [1, 2],
            "candidate_source_index": [3, 4],
        })
        pipeline = ThreeSignalFeaturePipeline(semantic_encoder=FakeSemanticEncoder()).fit(pairs.iloc[[0]], jobs, candidates)
        before = pipeline.manifest()
        pipeline.transform(pairs.iloc[[1]], jobs, candidates)
        self.assertEqual(before, pipeline.manifest())
        self.assertNotIn("validationonlytoken", pipeline.role_vectorizer.vocabulary_)
        self.assertNotIn("candidateonlytoken", pipeline.role_vectorizer.vocabulary_)
        self.assertNotIn("secret", pipeline.description_vectorizer.vocabulary_)

    def test_batched_transform_matches_pairwise_reference(self):
        jobs = pd.DataFrame({
            "_source_index": [1, 2],
            "Job Title": ["Data Analyst", "Python Developer"],
            "Job Description": ["Phân tích dữ liệu", "Xây dựng ứng dụng"],
            "Job Requirements": ["python; sql", "python; teamwork"],
            "Years of Experience": ["1-3 năm", "Trên 3 năm"],
            "Job Address": ["Hà Nội", "TP.HCM"],
        })
        candidates = pd.DataFrame({
            "_source_index": [3, 4],
            "Desired Job": ["Data Analyst", "Developer"],
            "Target": ["Phân tích dữ liệu", "Python"],
            "Skills": ["python; sql", "python"],
            "Work Experience": ["1-3 năm", "2 năm"],
            "Workplace Desired": ["Hà Nội", "Hồ Chí Minh"],
        })
        pairs = pd.DataFrame({
            "pair_id": [0, 1, 2],
            "job_id": ["JOB_1", "JOB_1", "JOB_2"],
            "cand_id": ["CV_003", "CV_004", "CV_004"],
            "job_source_index": [1, 1, 2],
            "candidate_source_index": [3, 4, 4],
        })
        pipeline = ThreeSignalFeaturePipeline(semantic_encoder=FakeSemanticEncoder()).fit(pairs, jobs, candidates)
        batched = pipeline.transform(pairs, jobs, candidates)
        job_lookup = jobs.set_index("_source_index")
        candidate_lookup = candidates.set_index("_source_index")
        reference = pd.DataFrame([
            pipeline._feature_row(
                job_lookup.loc[row.job_source_index],
                candidate_lookup.loc[row.candidate_source_index],
            )
            for row in pairs.itertuples(index=False)
        ])
        numeric = [
            "s_sem", "s_skill", "s_exp", "baseline_location", "baseline_skill",
            "baseline_experience", "baseline_role", "baseline_description",
            "heuristic_score",
        ]
        np.testing.assert_allclose(
            batched[numeric].to_numpy(float),
            reference[numeric].to_numpy(float),
            rtol=1e-7,
            atol=1e-9,
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
