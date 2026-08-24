import unittest

import numpy as np
import pandas as pd

from src.data.loader import FEATURE_COLS
from src.models.pairing import (
    build_pair_table,
    build_rank_pairs,
    build_rank_pairs_robust,
    pair_table_hash,
)
from src.models.skill_gap import build_gap_targets, parse_skill_set


class SkillGapTests(unittest.TestCase):
    def test_parser_removes_punctuation_only_tokens(self):
        self.assertEqual(
            parse_skill_set("python .. java ... .._• sql"),
            {"python", "java", "sql"},
        )

    def test_gap_vocabulary_contains_only_sanitized_tokens(self):
        frame = pd.DataFrame({
            "job_skills": ["python .. sql", "java ..."],
            "user_skills": ["python", ""],
        })

        targets, vocabulary = build_gap_targets(frame)

        self.assertEqual(vocabulary, ["java", "python", "sql"])
        self.assertEqual(targets.shape, (2, 3))
        self.assertTrue(np.isfinite(targets).all())


class PairingTests(unittest.TestCase):
    @staticmethod
    def frame(scores):
        rows = []
        for index, score in enumerate(scores):
            row = {col: float(index + j) for j, col in enumerate(FEATURE_COLS)}
            rows.append({
                "pair_id": index,
                "job_id": "JOB_1",
                "cand_id": f"CV_{index}",
                "y_prob": score,
                "lf_skill": -1 if index == len(scores) - 1 else 1,
                "lf_sem": 1 if index == 0 else 0,
                "lf_exp": 0,
                "lf_role": 0,
                "lf_loc": 0,
                **row,
            })
        return pd.DataFrame(rows)

    def test_fixed_delta_never_falls_back(self):
        frame = self.frame([0.50, 0.515])

        with self.assertRaisesRegex(ValueError, "protocol forbids fallback"):
            build_rank_pairs_robust(
                frame,
                margin=0.02,
                margin_fallbacks=(0.01, 0.0),
                min_pairs=1,
                rng=np.random.RandomState(1),
            )

    def test_ties_never_form_pairs(self):
        table, diagnostics = build_pair_table(
            self.frame([0.5, 0.5, 0.5]), pair_delta=0.0
        )

        self.assertTrue(table.empty)
        self.assertEqual(diagnostics.loc[0, "near_ties_removed"], 3)

    def test_pair_direction_is_strictly_higher_score_first(self):
        frame = self.frame([0.2, 0.8])
        left, right = build_rank_pairs(frame, margin=0.0)

        np.testing.assert_array_equal(left[0], frame.loc[1, FEATURE_COLS].to_numpy(float))
        np.testing.assert_array_equal(right[0], frame.loc[0, FEATURE_COLS].to_numpy(float))

    def test_pair_table_is_unique_within_job_and_respects_delta(self):
        table, _ = build_pair_table(
            self.frame([0.1, 0.3, 0.8]), pair_delta=0.2, max_pairs_per_job=10
        )

        self.assertEqual(len(table), 3)
        self.assertTrue((table["y_prob_delta"] >= 0.2).all())
        self.assertTrue(
            (table["preferred_y_prob"] > table["nonpreferred_y_prob"]).all()
        )
        unordered = table.apply(
            lambda row: tuple(sorted([
                row["preferred_row_id"], row["nonpreferred_row_id"]
            ])),
            axis=1,
        )
        self.assertFalse(unordered.duplicated().any())

    def test_hard_negatives_are_prioritized_deterministically(self):
        frame = self.frame([0.1, 0.2, 0.3, 0.9])
        first, _ = build_pair_table(
            frame, pair_delta=0.05, max_pairs_per_job=2, seed=7
        )
        second, _ = build_pair_table(
            frame, pair_delta=0.05, max_pairs_per_job=2, seed=7
        )

        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first["hard_negative"].all())
        self.assertEqual(pair_table_hash(first), pair_table_hash(second))


if __name__ == "__main__":
    unittest.main()
