import unittest

import numpy as np
import pandas as pd

from src.weak.aggregator import DEFAULT_LF_COLS, ProbabilisticLabelAggregator
from src.weak.pipeline import WeakLabelPipeline


class ProbabilisticLabelAggregatorTests(unittest.TestCase):
    def test_abstentions_are_excluded_from_each_lf_denominator(self):
        frame = pd.DataFrame({col: [0, 0, 0] for col in DEFAULT_LF_COLS})
        frame["lf_skill"] = [1, -1, 0]
        aggregator = ProbabilisticLabelAggregator(n_iter=1)
        aggregator._posterior = lambda *_: np.array([0.8, 0.2, 0.99])

        aggregator.fit(frame)

        self.assertAlmostEqual(aggregator.source_sensitivities[0], 0.8)
        self.assertAlmostEqual(aggregator.source_specificities[0], 0.8)

    def test_all_abstain_prediction_is_finite_and_equals_fitted_prior(self):
        train = pd.DataFrame({col: [1, -1, 0] for col in DEFAULT_LF_COLS})
        aggregator = ProbabilisticLabelAggregator().fit(train)
        abstain = pd.DataFrame({col: [0, 0] for col in DEFAULT_LF_COLS})

        probabilities = aggregator.predict_proba(abstain)

        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities, aggregator.p_prior)

    def test_prediction_does_not_mutate_fitted_parameters(self):
        train = pd.DataFrame({
            "lf_skill": [1, 1, -1, -1],
            "lf_sem": [1, 0, -1, 0],
            "lf_exp": [0, 1, 0, -1],
            "lf_role": [1, 0, -1, 0],
            "lf_loc": [0, 1, 0, -1],
        })
        validation = pd.DataFrame({col: [1, 0, -1] for col in DEFAULT_LF_COLS})
        aggregator = ProbabilisticLabelAggregator().fit(train)
        before = aggregator.parameters()

        aggregator.predict(validation)

        self.assertEqual(before, aggregator.parameters())

    def test_predict_before_fit_fails(self):
        frame = pd.DataFrame({col: [0] for col in DEFAULT_LF_COLS})
        with self.assertRaises(RuntimeError):
            ProbabilisticLabelAggregator().predict_proba(frame)


class WeakLabelPipelineTests(unittest.TestCase):
    @staticmethod
    def feature_frame(offset=0.0):
        return pd.DataFrame({
            "skill_iou": [0.0, 0.1 + offset, 0.4, 0.8],
            "desc_sem_sim": [0.05, 0.2 + offset, 0.5, 0.9],
            "role_match": [0.0, 0.15 + offset, 0.45, 0.85],
            "loc_match": [0.0, 1.0, 0.0, 1.0],
            "exp_score": [0.3, 0.7, 1.0, 1.0],
            "job_title": ["onsite role"] * 4,
        })

    def test_transform_before_fit_fails(self):
        with self.assertRaises(RuntimeError):
            WeakLabelPipeline().transform(self.feature_frame())

    def test_validation_batches_do_not_refit_label_model(self):
        pipeline = WeakLabelPipeline()
        pipeline.fit_transform_train(self.feature_frame(), method="dawid_skene")
        before = pipeline.aggregator_parameters()

        pipeline.transform(self.feature_frame(0.01), method="dawid_skene")
        middle = pipeline.aggregator_parameters()
        pipeline.transform(self.feature_frame(0.02), method="dawid_skene")

        self.assertEqual(before, middle)
        self.assertEqual(before, pipeline.aggregator_parameters())


if __name__ == "__main__":
    unittest.main()
