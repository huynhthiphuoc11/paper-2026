import copy
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.experiments.protocol import (
    ExperimentProtocol,
    create_immutable_run_directory,
    protocol_payload_sha256,
)
from src.experiments.runner import ThreeSignalExperiment


ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_test_cannot_open_before_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = ExperimentProtocol(
                Path(directory) / "gold_test_opened.json",
                protocol_sha256="a" * 64,
            )
            with self.assertRaises(RuntimeError):
                protocol.open_gold_test_once()

    def test_test_can_open_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "gold_test_opened.json"
            protocol = ExperimentProtocol(sentinel, protocol_sha256="a" * 64)
            protocol.lock(0.5, "pairwise", {})
            protocol.open_gold_test_once()
            with self.assertRaises(RuntimeError):
                protocol.open_gold_test_once()

    def test_new_protocol_instance_cannot_reopen_durable_sentinel(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "gold_test_opened.json"
            first = ExperimentProtocol(sentinel, protocol_sha256="a" * 64)
            first.lock(0.5, "pairwise", {})
            first.open_gold_test_once()

            second = ExperimentProtocol(sentinel, protocol_sha256="a" * 64)
            second.lock(0.5, "pairwise", {})
            with self.assertRaisesRegex(RuntimeError, "already been opened"):
                second.open_gold_test_once()

    def test_protocol_hash_directories_are_immutable_and_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_hash = protocol_payload_sha256({"version": 1})
            second_hash = protocol_payload_sha256({"version": 2})
            first = create_immutable_run_directory(root, first_hash)
            marker = first / "old-artifact.txt"
            marker.write_text("preserve", encoding="utf-8")
            second = create_immutable_run_directory(root, second_hash)

            self.assertNotEqual(first, second)
            self.assertTrue(marker.exists())
            repeated = create_immutable_run_directory(root, first_hash)
            third = create_immutable_run_directory(root, first_hash)
            self.assertEqual(repeated.name, f"{first_hash}-run-002")
            self.assertEqual(third.name, f"{first_hash}-run-003")
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_config_locks_fixed_threshold_and_batch_units(self):
        config = yaml.safe_load(
            (ROOT / "configs" / "experiment_3signal.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            config["protocol_version"],
            "three-signal-cuda-tie-aware-lf-2026-08-25",
        )
        self.assertEqual(
            config["weak_supervision"]["threshold_policy"],
            "negative-global-positive-tail-v1",
        )
        self.assertEqual(config["weak_supervision"]["posterior_threshold"], 0.5)
        self.assertNotIn("posterior_threshold_grid", config["weak_supervision"])
        self.assertEqual(
            set(config["models"]["batch_sizes"]),
            {"pointwise", "pairwise", "listwise"},
        )
        self.assertEqual(config["models"]["listwise_logit_epsilon"], 1e-4)
        self.assertEqual(config["models"]["device"], "cuda")
        self.assertEqual(config["weak_supervision"]["label_model"], {
            "n_iter": 100,
            "parameter_clip": [0.51, 0.99],
            "prior_clip": [0.05, 0.95],
            "initial_accuracy": 0.75,
            "convergence_tolerance": 1e-7,
        })

    def test_environment_manifest_is_written_before_gold_test(self):
        experiment = ThreeSignalExperiment.__new__(ThreeSignalExperiment)
        experiment.output_dir = Path(tempfile.mkdtemp())
        (experiment.output_dir / "audit").mkdir()
        experiment.settings = {"models": {"device": "cpu"}}
        experiment.device = __import__("torch").device("cpu")
        manifest = experiment._write_environment_manifest()
        saved = json.loads(
            (experiment.output_dir / "audit" / "environment_manifest.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(saved, manifest)
        self.assertEqual(manifest["device_policy"]["requested"], "cpu")
        self.assertEqual(manifest["device_policy"]["resolved"], "cpu")
        self.assertIn("python", manifest)
        self.assertIn("platform", manifest)
        self.assertIn("packages", manifest)
        for package in ["numpy", "pandas", "scipy", "scikit-learn", "torch", "PyYAML"]:
            self.assertIn(package, manifest["packages"])

    def test_development_test_is_diagnostic_only_after_selection(self):
        experiment = ThreeSignalExperiment.__new__(ThreeSignalExperiment)
        experiment.stage = "weak_supervision_fitted"
        with self.assertRaisesRegex(RuntimeError, "formulation selection"):
            experiment.evaluate_development_test_once()

        class Result:
            model = object()

        experiment.stage = "formulation_selected"
        experiment.selected_formulation = "pointwise"
        experiment.settings = {"seeds": [7], "gold": {"k_values": [5, 10]}}
        experiment.models = {7: {"pointwise": Result()}}
        experiment.development_test_weak = pd.DataFrame({
            "job_id": ["A", "A", "B", "B"],
            "y_prob": [0.9, 0.1, 0.8, 0.2],
        })
        experiment.output_dir = Path(tempfile.mkdtemp())
        (experiment.output_dir / "diagnostics").mkdir()
        (experiment.output_dir / "tables").mkdir()

        import src.experiments.runner as runner_module
        original = runner_module.predict_scores
        runner_module.predict_scores = lambda model, frame: frame["y_prob"].to_numpy()
        try:
            selected_before = experiment.selected_formulation
            models_before = copy.copy(experiment.models)
            diagnostic = experiment.evaluate_development_test_once()
        finally:
            runner_module.predict_scores = original
        self.assertEqual(experiment.selected_formulation, selected_before)
        self.assertEqual(experiment.models, models_before)
        self.assertEqual(
            diagnostic.iloc[0]["selection_role"],
            "diagnostic_only_after_selection",
        )

    def test_missing_ablation_checkpoint_metadata_blocks_test_open(self):
        experiment = ThreeSignalExperiment.__new__(ThreeSignalExperiment)
        experiment.stage = "protocol_locked"
        experiment.checkpoint_metadata = {"main": {}}

        class Guard:
            opened = False

            def open_gold_test_once(self):
                self.opened = True

        experiment.protocol = Guard()
        with self.assertRaisesRegex(RuntimeError, "ablation checkpoint"):
            experiment.evaluate_gold_test_once()
        self.assertFalse(experiment.protocol.opened)


if __name__ == "__main__":
    unittest.main()
