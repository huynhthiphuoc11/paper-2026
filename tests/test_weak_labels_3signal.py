import unittest

import numpy as np
import pandas as pd

from src.experiments import LF_COLUMNS
from src.experiments.weak_labels import (
    DawidSkeneThreeSource,
    PercentileLabelingFunctions,
    binary_quality,
    lf_diagnostics,
    strict_three_of_three,
)


class WeakLabelTests(unittest.TestCase):
    @staticmethod
    def signals():
        return pd.DataFrame({
            "s_sem": [0.0, 0.2, 0.6, 0.9],
            "s_skill": [0.0, 0.1, 0.5, 0.8],
            "s_exp": [0.2, 0.4, 0.8, 1.0],
            "sem_available": [True] * 4,
            "skill_available": [True] * 4,
            "exp_available": [True] * 4,
        })

    def test_thresholds_are_frozen(self):
        lfs = PercentileLabelingFunctions().fit(self.signals())
        before = dict(lfs.thresholds)
        lfs.transform(self.signals() * 1 if False else self.signals())
        self.assertEqual(before, lfs.thresholds)

    def test_missing_source_abstains(self):
        lfs = PercentileLabelingFunctions().fit(self.signals())
        frame = self.signals().iloc[[0]].copy()
        frame["sem_available"] = False
        frame["s_sem"] = np.nan
        result = lfs.transform(frame)
        self.assertEqual(result.iloc[0]["lf_sem"], 0)

    def test_sparse_zero_mass_uses_positive_tail_percentile(self):
        values = [0.0] * 8 + [0.1, 0.2, 0.4, 0.8]
        frame = pd.DataFrame({
            "s_sem": values,
            "s_skill": values,
            "s_exp": values,
            "sem_available": [True] * len(values),
            "skill_available": [True] * len(values),
            "exp_available": [True] * len(values),
        })
        lfs = PercentileLabelingFunctions().fit(frame)

        self.assertEqual(lfs.thresholds["s_skill"]["negative"], 0.0)
        self.assertGreater(lfs.thresholds["s_skill"]["positive"], 0.0)
        self.assertLess(lfs.thresholds["s_skill"]["positive"], max(values))
        self.assertEqual(lfs.threshold_diagnostics["s_skill"], {
            "n_active": 12,
            "n_at_or_below_negative": 8,
            "n_positive_tail": 4,
        })
        votes = lfs.transform(frame)["lf_skill"].to_numpy()
        self.assertTrue((votes[:8] == -1).all())
        self.assertEqual(votes[8], 0)
        self.assertEqual(votes[-1], 1)

    def test_constant_signal_is_rejected(self):
        frame = self.signals().copy()
        frame["s_skill"] = 0.0
        with self.assertRaisesRegex(ValueError, "s_skill has no variation"):
            PercentileLabelingFunctions().fit(frame)

    def test_unknown_threshold_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported threshold policy"):
            PercentileLabelingFunctions(threshold_policy="legacy")

    def test_dawid_skene_assumptions_are_named_and_reported(self):
        model = DawidSkeneThreeSource(
            n_iter=17,
            parameter_clip=(0.55, 0.97),
            prior_clip=(0.1, 0.9),
            initial_accuracy=0.8,
            convergence_tolerance=1e-6,
        )
        frame = pd.DataFrame({
            "lf_sem": [1, 1, -1, -1, 0],
            "lf_skill": [1, 0, -1, 0, 1],
            "lf_exp": [0, 1, 0, -1, -1],
        })
        parameters = model.fit(frame).parameters()
        self.assertEqual(parameters["assumptions"], {
            "n_iter": 17,
            "parameter_clip": [0.55, 0.97],
            "prior_clip": [0.1, 0.9],
            "initial_accuracy": 0.8,
            "convergence_tolerance": 1e-6,
        })

    def test_dawid_skene_rejects_invalid_assumptions(self):
        with self.assertRaises(ValueError):
            DawidSkeneThreeSource(parameter_clip=(0.4, 0.99))
        with self.assertRaises(ValueError):
            DawidSkeneThreeSource(prior_clip=(0.9, 0.1))
        with self.assertRaises(ValueError):
            DawidSkeneThreeSource(initial_accuracy=1.0)
        with self.assertRaises(ValueError):
            DawidSkeneThreeSource(convergence_tolerance=0.0)

    def test_dawid_skene_inference_is_immutable(self):
        frame = pd.DataFrame({
            "lf_sem": [1, 1, -1, -1, 0],
            "lf_skill": [1, 0, -1, 0, 1],
            "lf_exp": [0, 1, 0, -1, -1],
        })
        model = DawidSkeneThreeSource().fit(frame)
        before = model.parameters()
        probabilities = model.predict_proba(frame)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertEqual(before, model.parameters())

    def test_lf_diagnostics_use_only_joint_non_abstentions(self):
        frame = pd.DataFrame({
            "lf_sem": [1, -1, 0, 0, 0, 0],
            "lf_skill": [1, 1, 1, -1, 0, 0],
            "lf_exp": [0, 0, 1, -1, 1, -1],
        })
        stats, pairs = lf_diagnostics(frame)
        sem_stats = stats.query("lf == 'lf_sem'").iloc[0]
        self.assertEqual(sem_stats["n_positive"], 1)
        self.assertEqual(sem_stats["n_negative"], 1)
        self.assertEqual(sem_stats["n_abstain"], 4)
        row = pairs.query("lf_a == 'lf_sem' and lf_b == 'lf_skill'").iloc[0]
        self.assertAlmostEqual(row["joint_coverage"], 2 / 6)
        self.assertAlmostEqual(row["agreement"], 0.5)
        self.assertAlmostEqual(row["conflict"], 0.5)
        self.assertAlmostEqual(row["agreement"] + row["conflict"], 1.0)
        self.assertAlmostEqual(row["spearman"], 0.0)

    def test_lf_diagnostics_return_nan_without_joint_votes(self):
        frame = pd.DataFrame({
            "lf_sem": [1, -1, 0],
            "lf_skill": [0, 0, 1],
            "lf_exp": [1, -1, 1],
        })
        _, pairs = lf_diagnostics(frame)
        row = pairs.query("lf_a == 'lf_sem' and lf_b == 'lf_skill'").iloc[0]
        self.assertTrue(np.isnan(row["agreement"]))
        self.assertTrue(np.isnan(row["conflict"]))
        self.assertTrue(np.isnan(row["spearman"]))

    def test_strict_three_of_three_abstains_on_disagreement(self):
        frame = pd.DataFrame({
            LF_COLUMNS[0]: [1, -1, 1],
            LF_COLUMNS[1]: [1, -1, 1],
            LF_COLUMNS[2]: [1, -1, 0],
        })
        predictions = strict_three_of_three(frame)
        self.assertEqual(predictions[0], 1.0)
        self.assertEqual(predictions[1], 0.0)
        self.assertTrue(np.isnan(predictions[2]))

    def test_binary_quality_reports_confusion_counts(self):
        quality = binary_quality(
            np.array([1, 1, 0, 0]),
            np.array([1.0, 0.0, 1.0, np.nan]),
        )
        self.assertEqual(quality["n_positive"], 2)
        self.assertEqual(quality["n_predicted_positive"], 2)
        self.assertEqual(quality["tp"], 1)
        self.assertEqual(quality["fp"], 1)
        self.assertEqual(quality["fn"], 1)
        self.assertEqual(quality["tn"], 0)
        self.assertEqual(quality["n_covered"], 3)


if __name__ == "__main__":
    unittest.main()
