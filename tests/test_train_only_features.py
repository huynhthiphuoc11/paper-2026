import unittest

import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.feature_pipeline import (
    TrainOnlyFeaturePipeline,
    sample_raw_entities,
    split_pairs_by_job,
)


class TrainOnlyFeaturePipelineTests(unittest.TestCase):
    @staticmethod
    def entities():
        jobs = pd.DataFrame({
            "_source_index": [10, 20],
            "Job Title": ["trainrole", "validationonlytoken"],
            "Job Description": ["train description", "heldoutdescriptiontoken"],
            "Job Requirements": ["python sql", "rust secretjobskill"],
            "Years of Experience": ["2", "3"],
            "Job Address": ["hà nội", "hà nội"],
        })
        candidates = pd.DataFrame({
            "_source_index": [30, 40],
            "Desired Job": ["trainrole", "heldoutcandidateonlytoken"],
            "Target": ["backend", "secretprofiletoken"],
            "Skills": ["python", "secretcandidateskill"],
            "Work Experience": ["2", "1"],
            "Workplace Desired": ["hà nội", "hà nội"],
        })
        return jobs, candidates

    @staticmethod
    def pairs():
        return pd.DataFrame({
            "pair_id": [0, 1],
            "job_id": ["JOB_10", "JOB_20"],
            "cand_id": ["CV_030", "CV_040"],
            "job_source_index": [10, 20],
            "candidate_source_index": [30, 40],
        })

    def test_vectorizers_and_skill_filter_fit_train_only(self):
        jobs, candidates = self.entities()
        train = self.pairs().iloc[[0]]
        heldout = self.pairs().iloc[[1]]
        pipeline = TrainOnlyFeaturePipeline(df_threshold=0.4).fit(
            train, jobs, candidates
        )
        manifest_before = pipeline.manifest()

        pipeline.transform(heldout, jobs, candidates)
        manifest_after = pipeline.manifest()

        self.assertEqual(manifest_before, manifest_after)
        self.assertEqual(manifest_before["fit_job_ids"], ["JOB_10"])
        self.assertEqual(manifest_before["fit_candidate_ids"], ["CV_030"])
        role_vocab = pipeline.role_vectorizer.vocabulary_
        desc_vocab = pipeline.desc_vectorizer.vocabulary_
        self.assertNotIn("validationonlytoken", role_vocab)
        self.assertNotIn("heldoutcandidateonlytoken", role_vocab)
        self.assertNotIn("heldoutdescriptiontoken", desc_vocab)
        self.assertNotIn("secretprofiletoken", desc_vocab)
        self.assertNotIn("secretjobskill", pipeline.high_df_skills)
        self.assertNotIn("secretcandidateskill", pipeline.high_df_skills)

    def test_split_is_job_disjoint(self):
        pairs = pd.DataFrame({
            "job_id": [f"JOB_{index}" for index in range(10) for _ in range(2)]
        })
        train, validation, test, _ = split_pairs_by_job(
            pairs, seed=42, train_ratio=0.7, val_ratio=0.15
        )

        train_jobs = set(train["job_id"])
        validation_jobs = set(validation["job_id"])
        test_jobs = set(test["job_id"])
        self.assertTrue(train_jobs.isdisjoint(validation_jobs))
        self.assertTrue(train_jobs.isdisjoint(test_jobs))
        self.assertTrue(validation_jobs.isdisjoint(test_jobs))

    def test_sampling_excludes_all_gold_entities(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = pd.DataFrame({
                "Job Title": ["A", "B", "C"],
                "Job Requirements": ["x", "y", "z"],
            })
            candidates = pd.DataFrame({
                "Desired Job": ["A", "B", "C", "D"],
                "Skills": ["x", "y", "z", "w"],
            })
            jobs.to_csv(root / "JOB_DATA_FINAL.csv", index=False)
            candidates.to_csv(root / "USER_DATA_FINAL.csv", index=False)

            sampled = sample_raw_entities(
                str(root),
                n_jobs=2,
                n_candidates=3,
                candidates_per_job=2,
                seed=4,
                excluded_job_ids={"JOB_01"},
                excluded_candidate_ids={"CV_002"},
            )

            self.assertNotIn("JOB_01", set(sampled.pairs["job_id"]))
            self.assertNotIn("CV_002", set(sampled.pairs["cand_id"]))


if __name__ == "__main__":
    unittest.main()
