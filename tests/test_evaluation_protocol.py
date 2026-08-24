import unittest

import numpy as np
import pandas as pd

from src.eval.metrics import map_at_k, paired_job_bootstrap, per_job_ranking_metrics


class RankingMetricProtocolTests(unittest.TestCase):
    def test_map_uses_relevance_at_least_two(self):
        relevance = np.array([1, 2, 0])
        scores = np.array([0.9, 0.8, 0.1])

        self.assertAlmostEqual(map_at_k(relevance, scores, k=3), 0.5)

    def test_map_at_k_denominator_is_capped_at_k(self):
        relevance = np.array([2, 2, 2, 2])
        scores = np.array([0.9, 0.8, 0.7, 0.6])

        self.assertAlmostEqual(map_at_k(relevance, scores, k=2), 1.0)

    def test_per_job_metrics_keep_jobs_separate(self):
        frame = pd.DataFrame({
            "job_id": ["A", "A", "B", "B"],
            "relevance": [2, 0, 0, 2],
        })
        result = per_job_ranking_metrics(
            frame, scores=[0.9, 0.1, 0.9, 0.1], k_list=(1,)
        )

        self.assertEqual(result["job_id"].tolist(), ["A", "B"])
        self.assertEqual(result.loc[0, "map@1"], 1.0)
        self.assertEqual(result.loc[1, "map@1"], 0.0)

    def test_bootstrap_resamples_paired_job_differences(self):
        metrics = pd.DataFrame({
            "job_id": ["A", "B", "C", "A", "B", "C"],
            "model": ["H", "H", "H", "M1", "M1", "M1"],
            "ndcg@5": [0.1, 0.2, 0.3, 0.3, 0.4, 0.5],
        })

        result = paired_job_bootstrap(
            metrics, "H", "M1", "ndcg@5", n_bootstraps=200, seed=4
        )

        self.assertEqual(result["n_jobs"], 3)
        self.assertAlmostEqual(result["mean_delta"], 0.2)
        self.assertAlmostEqual(result["ci_95_low"], 0.2)
        self.assertAlmostEqual(result["ci_95_high"], 0.2)


if __name__ == "__main__":
    unittest.main()
