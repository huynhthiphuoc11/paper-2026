import unittest

import pandas as pd

from src.data.loader import FEATURE_COLS
from src.eval.perturbation import apply_isolated_perturbation


class PerturbationIsolationTests(unittest.TestCase):
    def setUp(self):
        self.row = pd.Series({
            "loc_match": 1.0,
            "skill_iou": 0.8,
            "exp_score": 0.9,
            "role_match": 0.7,
            "desc_sem_sim": 0.6,
        })

    def changed_features(self, kind):
        perturbed = apply_isolated_perturbation(self.row, kind)
        return {
            feature
            for feature in FEATURE_COLS
            if perturbed[feature] != self.row[feature]
        }

    def test_skill_perturbation_changes_only_skill(self):
        self.assertEqual(self.changed_features("skill"), {"skill_iou"})

    def test_experience_perturbation_changes_only_experience(self):
        self.assertEqual(self.changed_features("exp"), {"exp_score"})

    def test_domain_perturbation_changes_only_role_and_description(self):
        self.assertEqual(
            self.changed_features("domain"), {"role_match", "desc_sem_sim"}
        )


if __name__ == "__main__":
    unittest.main()
