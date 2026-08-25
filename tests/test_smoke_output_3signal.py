import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "three_signal" / "smoke"


class SmokeOutputTests(unittest.TestCase):
    def test_required_artifacts_exist(self):
        required = [
            "audit/raw_data_audit.json",
            "audit/gold_split_manifest.json",
            "audit/development_split_manifest.json",
            "audit/feature_manifest.json",
            "audit/weak_supervision_parameters.json",
            "audit/protocol_lock.json",
            "audit/protocol_final.json",
            "audit/run_manifest.json",
            "tables/label_quality_gold_validation.csv",
            "tables/formulation_selection.csv",
            "tables/gold_test_main_and_ablations.csv",
            "tables/paired_bootstrap_ci.csv",
            "predictions/gold_test_predictions.csv",
        ]
        self.assertEqual([name for name in required if not (OUTPUT / name).exists()], [])

    def test_protocol_lock_precedes_one_time_test_state(self):
        locked = json.loads((OUTPUT / "audit" / "protocol_lock.json").read_text(encoding="utf-8"))
        final = json.loads((OUTPUT / "audit" / "protocol_final.json").read_text(encoding="utf-8"))
        self.assertTrue(locked["locked"])
        self.assertFalse(locked["test_opened"])
        self.assertTrue(final["test_opened"])

    def test_gold_split_and_feature_leakage_contract(self):
        gold = json.loads((OUTPUT / "audit" / "gold_split_manifest.json").read_text(encoding="utf-8"))
        features = json.loads((OUTPUT / "audit" / "feature_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(gold["validation_jobs"]), 4)
        self.assertEqual(len(gold["test_jobs"]), 8)
        self.assertTrue(set(gold["validation_jobs"]).isdisjoint(gold["test_jobs"]))
        self.assertEqual(features["feature_columns"], ["s_sem", "s_skill", "s_exp"])
        self.assertTrue(set(features["fit_job_ids"]).isdisjoint(gold["validation_jobs"] + gold["test_jobs"]))

    def test_exactly_two_ablations_are_reported(self):
        summary = pd.read_csv(OUTPUT / "tables" / "gold_test_main_and_ablations.csv")
        systems = set(summary["system"])
        self.assertEqual(systems, {
            "manual_score_h", "selected_ltr",
            "ablation_mean_signal_ltr", "ablation_direct_probability",
        })
        bootstrap = pd.read_csv(OUTPUT / "tables" / "paired_bootstrap_ci.csv")
        self.assertEqual(set(bootstrap["comparison"]), {"main", "core_1", "core_2"})


if __name__ == "__main__":
    unittest.main()
