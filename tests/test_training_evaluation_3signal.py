import inspect
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from src.experiments import FEATURE_COLUMNS
from src.experiments.evaluation import (
    paired_job_bootstrap, per_job_metrics, reciprocal_rank, select_formulation,
)
from src.experiments.models import LinearScorer
from src.experiments.training import (
    _listwise_query_batches,
    _materialize_dataset,
    _pointwise_tensor_loss,
    _prepared_listwise_loss,
    _resolve_device,
    build_pair_table,
    listnet_target_distribution,
    listwise_loss,
    pair_table_hash,
    pairwise_loss,
    pointwise_loss,
    predict_scores,
    select_hyperparameters,
)


class TrainingEvaluationTests(unittest.TestCase):
    @staticmethod
    def frame():
        rows = []
        for job in ["A", "B"]:
            for index, probability in enumerate([0.1, 0.5, 0.9]):
                rows.append({
                    "pair_id": len(rows), "job_id": job, "cand_id": f"{job}{index}",
                    "y_prob": probability,
                    **{column: probability for column in FEATURE_COLUMNS},
                })
        return pd.DataFrame(rows)

    def test_pair_table_direction_delta_and_hash(self):
        frame = self.frame()
        first = build_pair_table(frame, 0.2, 2, 7)
        second = build_pair_table(frame, 0.2, 2, 7)
        self.assertTrue((first["delta"] >= 0.2).all())
        self.assertEqual(pair_table_hash(first), pair_table_hash(second))
        for row in first.itertuples():
            self.assertGreater(frame.loc[row.preferred_index, "y_prob"], frame.loc[row.nonpreferred_index, "y_prob"])

    def test_formulation_select_prefers_lower_pairwise_loss_in_tolerance(self):
        import pandas as pd
        summary = pd.DataFrame([
            {"formulation": "pointwise", "ndcg@5": 0.50, "validation_pairwise_loss": 0.90},
            {"formulation": "pointwise", "ndcg@5": 0.52, "validation_pairwise_loss": 0.90},
            {"formulation": "pairwise",  "ndcg@5": 0.50, "validation_pairwise_loss": 0.10},
            {"formulation": "pairwise",  "ndcg@5": 0.52, "validation_pairwise_loss": 0.10},
            {"formulation": "listwise",  "ndcg@5": 0.52, "validation_pairwise_loss": 0.05},
        ])
        self.assertEqual(select_formulation(summary, tie_tolerance=0.005), "listwise")
        self.assertEqual(select_formulation(summary, tie_tolerance=0.02), "listwise")

    def test_listnet_logit_targets_are_finite_ordered_and_not_flat(self):
        probabilities = torch.linspace(0.0, 1.0, 200)
        distribution = listnet_target_distribution(
            probabilities, temperature=1.0, epsilon=1e-4
        )
        self.assertTrue(torch.isfinite(distribution).all())
        self.assertAlmostEqual(float(distribution.sum()), 1.0, places=6)
        self.assertTrue(torch.all(distribution[1:] >= distribution[:-1]))
        self.assertGreater(float(distribution.max()), 10.0 / len(distribution))

    def test_listwise_loss_groups_queries(self):
        model = LinearScorer()
        loss = listwise_loss(
            model, self.frame(), temperature=1.0, logit_epsilon=1e-4
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_listwise_batch_size_counts_whole_queries(self):
        frame = pd.concat([
            self.frame(),
            self.frame().assign(job_id=lambda value: value["job_id"] + "2"),
        ], ignore_index=True)
        batches = list(_listwise_query_batches(frame, queries_per_batch=2))
        self.assertEqual([batch["job_id"].nunique() for batch in batches], [2, 2])
        self.assertEqual(sum(len(batch) for batch in batches), len(frame))

    def test_confirmed_pointwise_and_ranknet_losses_are_unchanged(self):
        pointwise_source = inspect.getsource(pointwise_loss)
        pairwise_source = inspect.getsource(pairwise_loss)
        self.assertIn("binary_cross_entropy_with_logits", pointwise_source)
        self.assertIn("frame[\"y_prob\"]", pointwise_source)
        self.assertIn("F.softplus", pairwise_source)
        self.assertIn("model(preferred) - model(nonpreferred)", pairwise_source)

    def test_materialized_pointwise_path_matches_dataframe_path(self):
        frame = self.frame()
        model = LinearScorer()
        prepared = _materialize_dataset(frame, device="cpu")
        self.assertEqual(prepared.features.device.type, "cpu")
        self.assertEqual(prepared.targets.device.type, "cpu")
        self.assertTrue(torch.allclose(
            pointwise_loss(model, frame),
            _pointwise_tensor_loss(model, prepared.features, prepared.targets),
        ))
        np.testing.assert_allclose(
            predict_scores(model, frame),
            predict_scores(model, prepared),
            rtol=1e-7,
            atol=1e-7,
        )

    def test_prepared_listwise_loss_matches_dataframe_path_and_covers_queries(self):
        frame = pd.concat([
            self.frame(),
            self.frame().iloc[:2].assign(job_id="C", pair_id=[20, 21]),
        ], ignore_index=True)
        model = LinearScorer()
        prepared = _materialize_dataset(frame, device="cpu")
        self.assertTrue(torch.allclose(
            listwise_loss(model, frame, temperature=1.0, logit_epsilon=1e-4),
            _prepared_listwise_loss(
                model, prepared, prepared.query_order,
                temperature=1.0, logit_epsilon=1e-4,
            ),
            rtol=1e-6,
            atol=1e-7,
        ))
        covered = torch.cat([prepared.query_indices[key] for key in prepared.query_order])
        self.assertEqual(sorted(covered.tolist()), list(range(len(frame))))
        self.assertEqual(len(torch.unique(covered)), len(frame))

    def test_pairwise_state_is_built_once_for_entire_grid(self):
        frame = self.frame()
        import src.experiments.training as training_module
        original = training_module.build_pair_table
        with mock.patch.object(
            training_module,
            "build_pair_table",
            wraps=original,
        ) as patched:
            _, grid = select_hyperparameters(
                "pairwise", frame, frame, seed=7,
                learning_rates=[0.01, 0.02], batch_sizes=[2, 4],
                max_epochs=1, patience=1, pair_delta=0.2,
                max_pairs_per_job=2, listwise_temperature=1.0,
                listwise_logit_epsilon=1e-4, device="cpu",
            )
        self.assertEqual(patched.call_count, 2)
        self.assertEqual(grid["train_pair_hash"].nunique(), 1)
        self.assertEqual(grid["validation_pair_hash"].nunique(), 1)

    def test_grid_materializes_each_split_once_and_records_cpu_device(self):
        frame = self.frame()
        import src.experiments.training as training_module
        original = training_module._materialize_dataset
        with mock.patch.object(
            training_module,
            "_materialize_dataset",
            wraps=original,
        ) as patched:
            result, grid = select_hyperparameters(
                "pointwise", frame, frame, seed=7,
                learning_rates=[0.01, 0.02], batch_sizes=[2, 4],
                max_epochs=1, patience=1, pair_delta=0.2,
                max_pairs_per_job=2, listwise_temperature=1.0,
                listwise_logit_epsilon=1e-4, device="cpu",
            )
        self.assertEqual(patched.call_count, 2)
        self.assertEqual(result.device, "cpu")
        self.assertEqual(set(grid["device"]), {"cpu"})
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in result.model.parameters()))

    def test_cuda_device_policy_fails_early_when_unavailable(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA.*not available"):
                _resolve_device("cuda")

    def test_cuda_device_policy_resolves_first_gpu(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.device_count", return_value=1),
        ):
            self.assertEqual(str(_resolve_device("cuda")), "cuda:0")
            self.assertEqual(str(_resolve_device("cuda:0")), "cuda:0")

    def test_cuda_device_policy_rejects_invalid_index(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.device_count", return_value=1),
        ):
            with self.assertRaisesRegex(ValueError, "index"):
                _resolve_device("cuda:1")

    def test_training_records_regularization_and_history(self):
        frame = self.frame()
        result, grid = select_hyperparameters(
            "pointwise", frame, frame, seed=7,
            learning_rates=[0.01], batch_sizes=[4], max_epochs=4, patience=2,
            pair_delta=0.2, max_pairs_per_job=2, listwise_temperature=1.0,
            listwise_logit_epsilon=1e-4,
            weight_decay=1e-4, gradient_clip_norm=5.0,
        )
        self.assertEqual(result.weight_decay, 1e-4)
        self.assertEqual(result.gradient_clip_norm, 5.0)
        self.assertEqual(len(result.history), result.epochs_ran)
        self.assertEqual(
            set(result.history.columns),
            {
                "epoch", "train_loss", "validation_loss", "generalization_gap",
                "validation_weak_ndcg_at_5",
            },
        )
        self.assertEqual(result.selection_metric, "validation_weak_ndcg_at_5")
        self.assertTrue(np.isfinite(result.best_validation_ndcg_at_5))
        self.assertIn("batch_unit", grid.columns)
        self.assertAlmostEqual(
            result.best_validation_loss - result.best_train_loss,
            result.metadata()["generalization_gap"],
        )
        self.assertIn("epochs_ran", grid.columns)

    def test_mrr_uses_grade_at_least_two(self):
        self.assertEqual(reciprocal_rank([1, 2, 0], [0.9, 0.8, 0.1]), 0.5)

    def test_metrics_are_per_job_and_bootstrap_is_paired(self):
        frame = pd.DataFrame({"job_id": ["A", "A", "B", "B"], "relevance": [3, 0, 0, 3]})
        metrics = per_job_metrics(frame, [1.0, 0.0, 1.0, 0.0], k_values=(1,))
        self.assertEqual(metrics["job_id"].tolist(), ["A", "B"])
        combined = pd.DataFrame({
            "system": ["H", "H", "M", "M"],
            "job_id": ["A", "B", "A", "B"],
            "ndcg@1": [0.0, 0.2, 0.2, 0.4],
        })
        result = paired_job_bootstrap(combined, "H", "M", "ndcg@1", 100, 4)
        self.assertAlmostEqual(result["mean_delta"], 0.2)

    def test_formulation_tie_breaks_on_common_pairwise_loss_then_simplicity(self):
        summary = pd.DataFrame({
            "formulation": ["pointwise", "pairwise", "listwise"],
            "ndcg@5": [0.800, 0.804, 0.803],
            "validation_pairwise_loss": [0.30, 0.20, 0.20],
        })
        self.assertEqual(select_formulation(summary), "pairwise")


if __name__ == "__main__":
    unittest.main()
