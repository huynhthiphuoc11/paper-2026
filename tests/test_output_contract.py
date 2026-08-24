import json
import unittest
from pathlib import Path

import pandas as pd


class SmokeOutputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1] / "results"

    def test_required_output_tree_exists(self):
        required = [
            "predictions/H.csv",
            "predictions/B1.csv",
            "predictions/B2.csv",
            "predictions/M1.csv",
            "predictions/M2.csv",
            "metrics/ranking_metrics.csv",
            "metrics/bootstrap_ci.csv",
            "metrics/perturbation_metrics.csv",
            "diagnostics/lf_statistics.csv",
            "diagnostics/lf_agreement.csv",
            "diagnostics/ds_statistics.csv",
            "diagnostics/pair_sampling.csv",
            "tables/main_results.csv",
            "tables/bootstrap_results.csv",
            "tables/perturbation_results.csv",
        ]
        self.assertEqual(
            [relative for relative in required if not (self.root / relative).exists()],
            [],
        )

    def test_prediction_schema_has_no_training_labels(self):
        resolved = json.loads(
            (self.root / "audit" / "resolved_config.json").read_text(
                encoding="utf-8"
            )
        )
        expected_rows = 100 * len(resolved["seeds"])
        for model in ["H", "B1", "B2", "M1", "M2"]:
            frame = pd.read_csv(self.root / "predictions" / f"{model}.csv")
            self.assertEqual(
                frame.columns.tolist(), ["seed", "job_id", "cv_id", "score"]
            )
            self.assertEqual(len(frame), expected_rows)

    def test_pair_hashes_prove_m1_m2_parity(self):
        hashes = pd.read_csv(self.root / "audit" / "pair_hashes.csv")
        self.assertTrue(
            (
                hashes["m1_train_pair_hash"]
                == hashes["m2_train_pair_hash"]
            ).all()
        )
        self.assertTrue(
            (
                hashes["m1_validation_pair_hash"]
                == hashes["m2_validation_pair_hash"]
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
