import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import FEATURE_COLS, RealKaggleDatasetAdapter


EXPECTED_FEATURES = [
    "loc_match",
    "skill_iou",
    "exp_score",
    "role_match",
    "desc_sem_sim",
]


class FeatureContractTests(unittest.TestCase):
    def test_model_feature_contract_is_exactly_five_columns(self):
        self.assertEqual(FEATURE_COLS, EXPECTED_FEATURES)

    def test_loader_keeps_diagnostics_outside_model_features(self):
        adapter = RealKaggleDatasetAdapter(data_dir="unused")
        adapter._extract_skills = lambda text: set(str(text).split())
        adapter._parse_experience = lambda text: float(text)
        adapter._check_loc_overlap = lambda user, job: user == job
        space = {
            "df_jobs": pd.DataFrame([{
                "Job Title": "Engineer",
                "Job Requirements": "python sql",
                "Years of Experience": "3",
                "Job Address": "hanoi",
            }]),
            "df_users": pd.DataFrame([{
                "Skills": "python",
                "Work Experience": "2",
                "Workplace Desired": "hanoi",
            }]),
            "role_sim_matrix": np.array([[0.5]]),
            "desc_sim_matrix": np.array([[0.4]]),
        }

        row = adapter._features_for_indices(0, 0, space)

        diagnostics = {
            "required_skill_match_ratio",
            "missing_required_skill_ratio",
            "experience_gap",
            "job_required_years",
            "cv_years",
        }
        self.assertTrue(diagnostics.issubset(row))
        self.assertTrue(diagnostics.isdisjoint(FEATURE_COLS))
        self.assertEqual(np.asarray([row[col] for col in FEATURE_COLS]).shape, (5,))


class LoaderScaleTests(unittest.TestCase):
    @staticmethod
    def adapter_with_stubbed_space(n_jobs=3, n_candidates=120):
        adapter = RealKaggleDatasetAdapter(data_dir="unused", random_seed=42)
        space = {
            "df_jobs": pd.DataFrame({"_source_index": range(1000, 1000 + n_jobs)}),
            "df_users": pd.DataFrame({"_source_index": range(2000, 2000 + n_candidates)}),
        }
        adapter.prepare_feature_space = lambda **_: space
        adapter._features_for_indices = lambda *_: {
            "job_title": "role",
            "loc_match": 1.0,
            "skill_iou": 0.5,
            "exp_score": 0.5,
            "experience_gap": 1.0,
            "job_required_years": 2.0,
            "cv_years": 1.0,
            "required_skill_match_ratio": 0.5,
            "missing_required_skill_ratio": 0.5,
            "role_match": 0.5,
            "desc_sem_sim": 0.5,
            "heuristic_score": 0.5,
            "heuristic_label": 1,
            "job_skills": "python",
            "user_skills": "python",
        }
        return adapter

    def test_candidates_per_job_controls_pair_count(self):
        adapter = self.adapter_with_stubbed_space()

        frame = adapter.load_and_preprocess(candidates_per_job=100)

        self.assertEqual(len(frame), 300)
        self.assertTrue((frame.groupby("job_id").size() == 100).all())
        self.assertEqual(set(frame["job_id"]), {"JOB_1000", "JOB_1001", "JOB_1002"})
        self.assertTrue(frame["cand_id"].str.match(r"CV_2\d{3}").all())

    def test_default_candidates_per_job_remains_seventy(self):
        adapter = self.adapter_with_stubbed_space()

        frame = adapter.load_and_preprocess()

        self.assertEqual(len(frame), 210)
        self.assertTrue((frame.groupby("job_id").size() == 70).all())

    def test_larger_sample_preserves_graded_identities_for_seed_42(self):
        root = Path(__file__).resolve().parents[1]
        jobs = pd.read_csv(root / "data" / "JOB_DATA_FINAL.csv").dropna(
            subset=["Job Title", "Job Requirements"]
        )
        users = pd.read_csv(root / "data" / "USER_DATA_FINAL.csv").dropna(
            subset=["Desired Job", "Skills"]
        )
        gold = pd.read_csv(
            root / "data" / "gold" / "human_validated_benchmark_graded_0_3.csv"
        )
        jobs_80 = jobs.assign(_source_index=jobs.index).sample(
            n=80, random_state=42
        ).reset_index(drop=True)
        jobs_160 = jobs.assign(_source_index=jobs.index).sample(
            n=160, random_state=42
        ).reset_index(drop=True)
        users_120 = users.assign(_source_index=users.index).sample(
            n=120, random_state=42
        ).reset_index(drop=True)
        users_240 = users.assign(_source_index=users.index).sample(
            n=240, random_state=42
        ).reset_index(drop=True)

        pd.testing.assert_series_equal(
            jobs_80["_source_index"],
            jobs_160.iloc[:80]["_source_index"].reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            users_120["_source_index"],
            users_240.iloc[:120]["_source_index"].reset_index(drop=True),
        )
        raw_jobs = pd.read_csv(root / "data" / "JOB_DATA_FINAL.csv")
        raw_users = pd.read_csv(root / "data" / "USER_DATA_FINAL.csv")
        for row in gold.itertuples(index=False):
            job_index = RealKaggleDatasetAdapter.parse_id_index(row.job_id)
            candidate_index = RealKaggleDatasetAdapter.parse_id_index(row.cand_id)
            self.assertEqual(row.job_title.strip(), raw_jobs.iloc[job_index]["Job Title"].strip())
            self.assertEqual(
                row.desired_job.strip(),
                raw_users.iloc[candidate_index]["Desired Job"].strip(),
            )


if __name__ == "__main__":
    unittest.main()
