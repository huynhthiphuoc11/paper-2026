from __future__ import annotations

import copy
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.experiments import FEATURE_COLUMNS, LF_COLUMNS
from src.experiments.data import (
    build_input_manifest,
    inter_annotator_agreement,
    load_gold_with_identity_check,
    load_raw_data,
    make_gold_split_manifest,
    sample_development_entities,
    split_development_pairs,
)
from src.experiments.evaluation import (
    macro_metrics,
    paired_job_bootstrap,
    per_job_metrics,
    select_formulation,
)
from src.experiments.features import ThreeSignalFeaturePipeline
from src.experiments.protocol import (
    ExperimentProtocol,
    create_immutable_run_directory,
    protocol_payload_sha256,
)
from src.experiments.training import (
    _resolve_device,
    _pair_tensors,
    build_pair_table,
    pairwise_loss,
    predict_scores,
    select_hyperparameters,
)
from src.experiments.utils import sha256_file, write_json
from src.experiments.weak_labels import (
    binary_quality,
    lf_diagnostics,
    strict_three_of_three,
    ThreeSourceWeakLabelPipeline,
)


class ThreeSignalExperiment:
    """Stage-oriented runner used by the executable notebook."""

    def __init__(self, config_path: str | Path, smoke: bool = False):
        self.root = Path(__file__).resolve().parents[2]
        self.config_path = self._resolve(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.smoke = bool(smoke)
        self.settings = self._resolve_settings(self.config, self.smoke)
        self.device = _resolve_device(self.settings["models"]["device"])
        self.data_root = self._resolve(self.settings["data_dir"]).resolve()
        self.gold_path = self._resolve(self.settings["gold_path"]).resolve()
        annotation_value = self.settings["gold"].get("independent_annotations_path")
        self.independent_annotations_path = (
            self._resolve(annotation_value).resolve() if annotation_value else None
        )
        mode = "smoke" if self.smoke else "full"
        self.protocol_payload = self._build_protocol_payload(mode)
        self.protocol_sha256 = protocol_payload_sha256(self.protocol_payload)
        output_root = self._resolve(self.settings["output_dir"]) / mode
        self.output_dir = create_immutable_run_directory(
            output_root, self.protocol_sha256
        )
        for name in ["audit", "diagnostics", "tables", "predictions", "checkpoints"]:
            (self.output_dir / name).mkdir(exist_ok=True)
        self.protocol = ExperimentProtocol(
            self.output_dir / "audit" / "gold_test_opened.json",
            protocol_sha256=self.protocol_sha256,
        )
        self.environment_manifest = self._write_environment_manifest()
        self.models: dict[int, dict[str, object]] = {}
        self.ablation_models: dict[int, object] = {}
        self.stage = "initialized"

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _write_environment_manifest(self) -> dict:
        packages = {}
        for distribution in [
            "numpy", "pandas", "scipy", "scikit-learn", "torch", "PyYAML",
            "sentence-transformers", "transformers", "matplotlib", "jupyter",
            "nbconvert", "ipykernel",
        ]:
            try:
                packages[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                packages[distribution] = None
        manifest = {
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "device_policy": {
                "requested": str(self.settings["models"]["device"]),
                "resolved": str(self.device),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_runtime": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_name": (
                    torch.cuda.get_device_name(self.device)
                    if self.device.type == "cuda"
                    else None
                ),
            },
            "packages": packages,
        }
        write_json(
            self.output_dir / "audit" / "environment_manifest.json",
            manifest,
        )
        return manifest

    @staticmethod
    def _resolve_settings(config: dict, smoke: bool) -> dict:
        settings = copy.deepcopy(config)
        if smoke:
            for key in ["n_jobs", "n_candidates", "candidates_per_job"]:
                settings["sample"][key] = settings["smoke"][key]
            settings["seeds"] = settings["smoke"]["seeds"]
            settings["models"]["learning_rates"] = settings["smoke"][
                "learning_rates"
            ]
            settings["models"]["batch_sizes"] = copy.deepcopy(
                settings["smoke"]["batch_sizes"]
            )
            for key in ["max_epochs", "patience"]:
                settings["models"][key] = settings["smoke"][key]
            settings["bootstrap"]["n_resamples"] = settings["smoke"]["n_resamples"]
        return settings

    def _build_protocol_payload(self, mode: str) -> dict:
        input_manifest = build_input_manifest(self.data_root)
        return {
            "protocol_version": self.settings["protocol_version"],
            "mode": mode,
            "settings": self.settings,
            "gold_split_policy": "sorted-job-id-seeded-shuffle-v1",
            "feature_definition": "three-signal-multilingual-embedding-v4",
            "model_selection_policy": "weak-validation-ndcg5-v1",
            "weak_label_definition": "negative-global-positive-tail-v1",
            "posterior_threshold_policy": "fixed-0.5-v1",
            "training_device_policy": "configured-device-required-v1",
            "input_files": input_manifest,
            "input_hashes": {
                "jobs": input_manifest["jobs"]["sha256"],
                "candidates": input_manifest["candidates"]["sha256"],
                "gold": sha256_file(self.gold_path),
                "independent_annotations": (
                    sha256_file(self.independent_annotations_path)
                    if self.independent_annotations_path
                    and self.independent_annotations_path.is_file()
                    else None
                ),
                "config": sha256_file(self.config_path),
            },
        }

    def audit_data_and_gold(self) -> tuple[pd.DataFrame, dict]:
        self.raw = load_raw_data(self.data_root)
        self.gold = load_gold_with_identity_check(self.gold_path, self.raw)
        self.gold_manifest = make_gold_split_manifest(
            self.gold,
            n_validation_jobs=int(self.settings["gold"]["validation_jobs"]),
            seed=int(self.settings["gold"]["split_seed"]),
        )
        split_path = (
            self.root / "data" / "splits" /
            "gold_split_manifest_label_blind_v3.json"
        )
        if split_path.exists():
            existing = json.loads(split_path.read_text(encoding="utf-8"))
            if existing != self.gold_manifest:
                raise RuntimeError(
                    "Existing label-blind Gold split manifest differs from protocol"
                )
        else:
            write_json(split_path, self.gold_manifest)
        self.gold_validation = self.gold[self.gold["job_id"].isin(self.gold_manifest["validation_jobs"])].reset_index(drop=True)
        self._gold_test_private = self.gold[self.gold["job_id"].isin(self.gold_manifest["test_jobs"])].reset_index(drop=True)
        self.iaa_audit = inter_annotator_agreement(
            self.independent_annotations_path,
            self.gold,
        )
        self.stage = "gold_audited"
        write_json(self.output_dir / "audit" / "raw_data_audit.json", self.raw.audit)
        write_json(self.output_dir / "audit" / "gold_split_manifest.json", self.gold_manifest)
        write_json(
            self.output_dir / "audit" / "inter_annotator_agreement.json",
            self.iaa_audit,
        )
        return self.gold_validation.copy(), dict(self.raw.audit)

    def prepare_development_data(self) -> dict:
        if self.stage != "gold_audited":
            raise RuntimeError("Audit Gold before development preparation")
        # Exclude every annotated entity, not merely Gold-test, from weak supervision.
        sampled = sample_development_entities(
            self.raw,
            n_jobs=int(self.settings["sample"]["n_jobs"]),
            n_candidates=int(self.settings["sample"]["n_candidates"]),
            candidates_per_job=int(self.settings["sample"]["candidates_per_job"]),
            seed=int(self.settings["seed"]),
            excluded_job_ids=set(self.gold["job_id"]),
            excluded_candidate_ids=set(self.gold["cand_id"]),
        )
        train_pairs, validation_pairs, test_pairs, self.development_manifest = split_development_pairs(
            sampled.pairs,
            train_ratio=float(self.settings["split"]["train_ratio"]),
            validation_ratio=float(self.settings["split"]["validation_ratio"]),
            seed=int(self.settings["split"]["seed"]),
        )
        self.feature_pipeline = ThreeSignalFeaturePipeline(
            semantic_model_name=str(self.settings["features"]["semantic_model_name"]),
            semantic_batch_size=int(self.settings["features"]["semantic_batch_size"]),
            semantic_device=str(self.device),
            semantic_max_features=int(self.settings["features"]["semantic_max_features"]),
            role_max_features=int(self.settings["features"]["role_max_features"]),
        ).fit(train_pairs, sampled.jobs, sampled.candidates)
        self.train_raw_features = self.feature_pipeline.transform(train_pairs, sampled.jobs, sampled.candidates)
        self.validation_raw_features = self.feature_pipeline.transform(validation_pairs, sampled.jobs, sampled.candidates)
        self.development_test_raw_features = self.feature_pipeline.transform(test_pairs, sampled.jobs, sampled.candidates)
        self.gold_validation_raw_features = self.feature_pipeline.transform_gold(self.gold_validation, self.raw)
        self.train, self.fill_values = self.feature_pipeline.impute_for_models(self.train_raw_features)
        self.validation, _ = self.feature_pipeline.impute_for_models(self.validation_raw_features, self.fill_values)
        self.development_test, _ = self.feature_pipeline.impute_for_models(self.development_test_raw_features, self.fill_values)
        self.gold_validation_features, _ = self.feature_pipeline.impute_for_models(self.gold_validation_raw_features, self.fill_values)
        feature_manifest = self.feature_pipeline.manifest()
        all_gold_jobs = set(self.gold["job_id"])
        all_gold_candidates = set(self.gold["cand_id"])
        if set(feature_manifest["fit_job_ids"]) & all_gold_jobs:
            raise AssertionError("Annotated job leaked into feature fit")
        if set(feature_manifest["fit_candidate_ids"]) & all_gold_candidates:
            raise AssertionError("Annotated candidate leaked into feature fit")
        self.sampled = sampled
        self.stage = "development_prepared"
        write_json(self.output_dir / "audit" / "development_split_manifest.json", self.development_manifest)
        write_json(self.output_dir / "audit" / "feature_manifest.json", feature_manifest)
        write_json(self.output_dir / "audit" / "imputation_values.json", self.fill_values)
        return {
            "sampled_jobs": len(sampled.jobs),
            "sampled_candidates": len(sampled.candidates),
            "sampled_pairs": len(sampled.pairs),
            "train_pairs": len(self.train),
            "validation_pairs": len(self.validation),
            "development_test_pairs": len(self.development_test),
        }

    def fit_weak_supervision(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.stage != "development_prepared":
            raise RuntimeError("Prepare development data before weak supervision")
        weak_config = self.settings["weak_supervision"]
        self.weak_pipeline = ThreeSourceWeakLabelPipeline(
            negative_percentile=float(weak_config["negative_percentile"]),
            positive_percentile=float(weak_config["positive_percentile"]),
            threshold_policy=str(weak_config["threshold_policy"]),
            label_model_config=copy.deepcopy(weak_config["label_model"]),
        )
        self.train_weak = self.weak_pipeline.fit_transform_train(self.train)
        parameters_before = self.weak_pipeline.parameters()
        self.validation_weak = self.weak_pipeline.transform(self.validation)
        self.development_test_weak = self.weak_pipeline.transform(self.development_test)
        self.gold_validation_weak = self.weak_pipeline.transform(self.gold_validation_features)
        if parameters_before != self.weak_pipeline.parameters():
            raise AssertionError("Held-out inference mutated weak supervision")
        lf_stats, lf_pairs = lf_diagnostics(self.train_weak)
        if not lf_pairs.empty and lf_pairs["spearman"].abs().max() >= float(weak_config["max_abs_spearman"]):
            raise RuntimeError("LF marginal-dependence diagnostic exceeded the predeclared threshold")
        gold_truth = (self.gold_validation_weak["relevance"].to_numpy(int) >= int(self.settings["gold"]["binary_threshold"])).astype(int)
        consensus_predictions = strict_three_of_three(self.gold_validation_weak)
        consensus_quality = binary_quality(gold_truth, consensus_predictions)
        self.posterior_threshold = float(weak_config["posterior_threshold"])
        threshold_table = pd.DataFrame([{
            "threshold": self.posterior_threshold,
            "policy": "fixed-0.5",
            "selection_data": "none",
        }])
        label_model_quality = binary_quality(
            gold_truth,
            (self.gold_validation_weak["y_prob"].to_numpy(float) >= self.posterior_threshold).astype(float),
        )
        label_condition_passed = bool(
            label_model_quality["recall"] > consensus_quality["recall"]
            and label_model_quality["precision"]
            >= consensus_quality["precision"] - 0.02
        )
        self.label_quality = pd.DataFrame([
            {"method": "strict_3_of_3", **consensus_quality},
            {
                "method": "dawid_skene", "threshold": self.posterior_threshold,
                **label_model_quality, "condition_passed": label_condition_passed,
            },
        ])
        # Persist audit before the confirmatory gate so a failed prerequisite
        # still leaves Table 4 and diagnostics on disk for the report.
        lf_stats.to_csv(self.output_dir / "diagnostics" / "lf_statistics.csv", index=False)
        lf_pairs.to_csv(self.output_dir / "diagnostics" / "lf_pair_diagnostics.csv", index=False)
        threshold_table.to_csv(self.output_dir / "diagnostics" / "posterior_threshold_selection.csv", index=False)
        self.label_quality.to_csv(self.output_dir / "tables" / "label_quality_gold_validation.csv", index=False)
        write_json(self.output_dir / "audit" / "weak_supervision_parameters.json", parameters_before)
        self.train_weak[["pair_id", "job_id", "cand_id", *LF_COLUMNS, "y_prob"]].to_csv(
            self.output_dir / "audit" / "train_weak_labels.csv", index=False
        )
        gold_validation_diagnostics = self.gold_validation_weak[
            ["pair_id", "job_id", "cand_id", "relevance", *LF_COLUMNS, "y_prob"]
        ].copy()
        gold_validation_diagnostics["binary_relevance"] = gold_truth
        gold_validation_diagnostics["dawid_skene_prediction"] = (
            gold_validation_diagnostics["y_prob"] >= self.posterior_threshold
        ).astype(int)
        gold_validation_diagnostics.to_csv(
            self.output_dir / "diagnostics" /
            "gold_validation_weak_label_diagnostics.csv",
            index=False,
        )
        self.label_gate_passed = label_condition_passed
        self.label_gate_reason = (
            None
            if label_condition_passed
            else (
                "Dawid-Skene did not achieve strictly higher recall than strict 3/3 "
                "while keeping precision within the predeclared 0.02 margin"
            )
        )
        self.stage = (
            "weak_supervision_fitted"
            if label_condition_passed or self.smoke
            else "confirmatory_blocked"
        )
        return self.label_quality.copy(), lf_stats, lf_pairs

    def train_and_select_formulation(self) -> pd.DataFrame:
        if self.stage != "weak_supervision_fitted":
            raise RuntimeError("Fit weak supervision before rankers")
        model_config = self.settings["models"]
        validation_rows = []
        selection_rows = []
        history_rows = []
        for seed_value in self.settings["seeds"]:
            seed = int(seed_value)
            self.models[seed] = {}
            for formulation in ["pointwise", "pairwise", "listwise"]:
                result, grid = select_hyperparameters(
                    formulation,
                    self.train_weak,
                    self.validation_weak,
                    seed,
                    model_config["learning_rates"],
                    model_config["batch_sizes"][formulation],
                    int(model_config["max_epochs"]),
                    int(model_config["patience"]),
                    float(model_config["pair_delta"]),
                    int(model_config["max_pairs_per_job"]),
                    float(model_config["listwise_temperature"]),
                    float(model_config["listwise_logit_epsilon"]),
                    float(model_config["weight_decay"]),
                    float(model_config["gradient_clip_norm"]),
                    str(self.device),
                )
                self.models[seed][formulation] = result
                for record in grid.to_dict("records"):
                    selection_rows.append({"seed": seed, **record})
                history = result.history.copy()
                history.insert(0, "seed", seed)
                history.insert(1, "formulation", formulation)
                history.insert(2, "learning_rate", result.learning_rate)
                history.insert(3, "batch_size", result.batch_size)
                history_rows.append(history)
                scores = predict_scores(result.model, self.gold_validation_weak)
                validation_pair_table = build_pair_table(
                    self.validation_weak,
                    float(model_config["pair_delta"]),
                    int(model_config["max_pairs_per_job"]),
                    seed,
                )
                preferred, nonpreferred = _pair_tensors(
                    self.validation_weak, validation_pair_table, str(self.device)
                )
                result.model.eval()
                with torch.no_grad():
                    validation_pairwise_loss = float(
                        pairwise_loss(result.model, preferred, nonpreferred).item()
                    )
                metrics = per_job_metrics(
                    self.gold_validation_weak,
                    scores,
                    target_column=self.settings["gold"]["relevance_column"],
                    k_values=tuple(self.settings["gold"]["k_values"]),
                )
                metrics.insert(0, "seed", seed)
                metrics.insert(1, "formulation", formulation)
                metrics.insert(2, "validation_pairwise_loss", validation_pairwise_loss)
                validation_rows.append(metrics)
                torch.save(
                    result.model.state_dict(),
                    self.output_dir / "checkpoints" / f"{formulation}_seed{seed}.pt",
                )
        self.formulation_validation = pd.concat(validation_rows, ignore_index=True)
        self.selected_formulation = select_formulation(
            self.formulation_validation,
            tie_tolerance=float(model_config["formulation_tie_tolerance"]),
        )
        self.formulation_summary = self.formulation_validation.groupby("formulation", as_index=False)[["ndcg@5", "ndcg@10", "mrr"]].mean()
        self.formulation_summary["selected"] = self.formulation_summary["formulation"] == self.selected_formulation
        self.training_history = pd.concat(history_rows, ignore_index=True)
        self.overfitting_diagnostics = pd.DataFrame(selection_rows)[[
            "seed", "formulation", "learning_rate", "batch_size", "batch_unit",
            "device", "best_epoch", "epochs_ran", "selection_metric",
            "best_validation_ndcg_at_5", "best_train_loss",
            "best_validation_loss", "generalization_gap", "stopped_early",
            "weight_decay", "gradient_clip_norm",
        ]]
        self.stage = "formulation_selected"
        self.formulation_validation.to_csv(self.output_dir / "diagnostics" / "formulation_validation_per_job.csv", index=False)
        self.formulation_summary.to_csv(self.output_dir / "tables" / "formulation_selection.csv", index=False)
        self.training_history.to_csv(self.output_dir / "diagnostics" / "training_history.csv", index=False)
        self.overfitting_diagnostics.to_csv(
            self.output_dir / "diagnostics" / "overfitting_diagnostics.csv", index=False
        )
        pd.DataFrame(selection_rows).to_csv(self.output_dir / "audit" / "model_selection_grid.csv", index=False)
        return self.formulation_summary.copy()

    def evaluate_development_test_once(self) -> pd.DataFrame:
        if self.stage != "formulation_selected":
            raise RuntimeError(
                "Development-test diagnostic requires completed formulation selection"
            )
        if hasattr(self, "development_test_diagnostic"):
            raise RuntimeError("Development-test diagnostic has already been evaluated")
        selected_before = self.selected_formulation
        checkpoint_ids_before = {
            seed: id(self.models[int(seed)][self.selected_formulation])
            for seed in self.settings["seeds"]
        }
        rows = []
        for seed_value in self.settings["seeds"]:
            seed = int(seed_value)
            result = self.models[seed][self.selected_formulation]
            scores = predict_scores(result.model, self.development_test_weak)
            metrics = per_job_metrics(
                self.development_test_weak,
                scores,
                target_column="y_prob",
                k_values=tuple(self.settings["gold"]["k_values"]),
            ).drop(columns="mrr")
            metrics.insert(0, "seed", seed)
            rows.append(metrics)
        self.development_test_per_job = pd.concat(rows, ignore_index=True)
        macro = self.development_test_per_job.groupby("seed", as_index=False)[
            ["ndcg@5", "ndcg@10"]
        ].mean()
        macro.insert(1, "formulation", self.selected_formulation)
        macro["selection_role"] = "diagnostic_only_after_selection"
        self.development_test_diagnostic = macro
        if self.selected_formulation != selected_before:
            raise AssertionError("Development-test diagnostic changed formulation")
        checkpoint_ids_after = {
            seed: id(self.models[int(seed)][self.selected_formulation])
            for seed in self.settings["seeds"]
        }
        if checkpoint_ids_after != checkpoint_ids_before:
            raise AssertionError("Development-test diagnostic changed selected checkpoints")
        self.development_test_per_job.to_csv(
            self.output_dir / "diagnostics" /
            "development_test_weak_ranking_per_job.csv",
            index=False,
        )
        self.development_test_diagnostic.to_csv(
            self.output_dir / "tables" /
            "development_test_weak_ranking_diagnostic.csv",
            index=False,
        )
        return self.development_test_diagnostic.copy()

    def _train_ablation_mean_signal(self, seed: int):
        train = self.train_weak.copy()
        validation = self.validation_weak.copy()
        train["y_prob"] = train[FEATURE_COLUMNS].mean(axis=1)
        validation["y_prob"] = validation[FEATURE_COLUMNS].mean(axis=1)
        result, _ = select_hyperparameters(
            self.selected_formulation,
            train,
            validation,
            seed,
            self.settings["models"]["learning_rates"],
            self.settings["models"]["batch_sizes"][self.selected_formulation],
            int(self.settings["models"]["max_epochs"]),
            int(self.settings["models"]["patience"]),
            float(self.settings["models"]["pair_delta"]),
            int(self.settings["models"]["max_pairs_per_job"]),
            float(self.settings["models"]["listwise_temperature"]),
            float(self.settings["models"]["listwise_logit_epsilon"]),
            float(self.settings["models"]["weight_decay"]),
            float(self.settings["models"]["gradient_clip_norm"]),
            str(self.device),
        )
        return result

    def prepare_confirmatory_checkpoints(self) -> dict:
        if self.stage != "formulation_selected":
            raise RuntimeError("Select formulation before preparing checkpoints")
        if not hasattr(self, "development_test_diagnostic"):
            raise RuntimeError(
                "Run the diagnostic-only development-test check before checkpoints"
            )
        main_metadata = {}
        ablation_metadata = {}
        for seed_value in self.settings["seeds"]:
            seed = int(seed_value)
            main_path = (
                self.output_dir / "checkpoints" /
                f"{self.selected_formulation}_seed{seed}.pt"
            )
            if not main_path.exists():
                raise RuntimeError(f"Missing selected-model checkpoint: {main_path}")
            main_metadata[str(seed)] = {
                "seed": seed,
                "formulation": self.selected_formulation,
                "path": str(main_path.relative_to(self.output_dir)),
                "sha256": sha256_file(main_path),
                "training": self.models[seed][self.selected_formulation].metadata(),
            }

            ablation = self._train_ablation_mean_signal(seed)
            self.ablation_models[seed] = ablation
            ablation_path = (
                self.output_dir / "checkpoints" /
                f"ablation_mean_signal_{self.selected_formulation}_seed{seed}.pt"
            )
            torch.save(ablation.model.state_dict(), ablation_path)
            ablation_metadata[str(seed)] = {
                "seed": seed,
                "formulation": self.selected_formulation,
                "path": str(ablation_path.relative_to(self.output_dir)),
                "sha256": sha256_file(ablation_path),
                "training": ablation.metadata(),
            }
        self.checkpoint_metadata = {
            "main": main_metadata,
            "ablation_mean_signal": ablation_metadata,
        }
        self.stage = "confirmatory_checkpoints_prepared"
        write_json(
            self.output_dir / "audit" / "confirmatory_checkpoints.json",
            self.checkpoint_metadata,
        )
        return copy.deepcopy(self.checkpoint_metadata)

    def _validate_checkpoint_metadata(self) -> None:
        metadata = getattr(self, "checkpoint_metadata", None)
        if not isinstance(metadata, dict) or not metadata.get("ablation_mean_signal"):
            raise RuntimeError(
                "Complete ablation checkpoint metadata is required before Gold-test"
            )
        expected_seeds = {str(int(value)) for value in self.settings["seeds"]}
        for family in ["main", "ablation_mean_signal"]:
            records = metadata.get(family)
            if not isinstance(records, dict) or set(records) != expected_seeds:
                raise RuntimeError(
                    f"Complete {family} checkpoint metadata is required before Gold-test"
                )
            for record in records.values():
                path = self.output_dir / record["path"]
                if not path.is_file() or sha256_file(path) != record.get("sha256"):
                    raise RuntimeError(
                        f"Locked {family} checkpoint is missing or hash-mismatched"
                    )

    def lock_protocol(self) -> dict:
        if self.stage != "confirmatory_checkpoints_prepared":
            raise RuntimeError("Prepare all confirmatory checkpoints before protocol lock")
        self._validate_checkpoint_metadata()
        split_path = (
            self.root / "data" / "splits" /
            "gold_split_manifest_label_blind_v3.json"
        )
        manifest = self.protocol.lock(
            self.posterior_threshold,
            self.selected_formulation,
            {
                "seeds": [int(value) for value in self.settings["seeds"]],
                "feature_manifest": self.feature_pipeline.manifest(),
                "weak_parameters": self.weak_pipeline.parameters(),
                "threshold_policy": "fixed-0.5",
                "training_device_policy": str(self.device),
                "gold_validation_uses": [
                    "label_quality_at_fixed_threshold",
                    "formulation_selection",
                ],
                "development_test": {
                    "selection_role": "diagnostic_only_after_selection",
                    "summary": self.development_test_diagnostic.to_dict("records"),
                },
                "confirmatory_checkpoints": self.checkpoint_metadata,
                "environment_manifest": self.environment_manifest,
                "environment_manifest_sha256": sha256_file(
                    self.output_dir / "audit" / "environment_manifest.json"
                ),
                "gold_validation_jobs": self.gold_manifest["validation_jobs"],
                "gold_test_jobs_sha256": sha256_file(split_path),
                "protocol_payload": self.protocol_payload,
            },
        )
        self.stage = "protocol_locked"
        write_json(self.output_dir / "audit" / "protocol_lock.json", manifest)
        return manifest

    def evaluate_gold_test_once(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.stage != "protocol_locked":
            raise RuntimeError("Protocol must be locked before Gold-test")
        self._validate_checkpoint_metadata()
        locked_checkpoints = self.protocol.lock_metadata.get(
            "confirmatory_checkpoints"
        )
        if locked_checkpoints != self.checkpoint_metadata:
            raise RuntimeError(
                "Protocol lock does not contain the current ablation checkpoint metadata"
            )
        self.protocol.open_gold_test_once()
        # Gold-test is transformed and weak-labeled only after the one-time gate opens.
        gold_test_raw = self.feature_pipeline.transform_gold(self._gold_test_private, self.raw)
        gold_test_features, _ = self.feature_pipeline.impute_for_models(gold_test_raw, self.fill_values)
        parameters_before_test = self.weak_pipeline.parameters()
        test = self.weak_pipeline.transform(gold_test_features)
        if parameters_before_test != self.weak_pipeline.parameters():
            raise AssertionError("Gold-test inference mutated weak supervision")
        per_job_rows = []
        prediction_rows = []
        for seed_value in self.settings["seeds"]:
            seed = int(seed_value)
            selected_model = copy.deepcopy(
                self.models[seed][self.selected_formulation].model
            ).to(self.device)
            selected_record = self.checkpoint_metadata["main"][str(seed)]
            selected_model.load_state_dict(torch.load(
                self.output_dir / selected_record["path"],
                map_location=self.device,
                weights_only=True,
            ))
            ablation_model = copy.deepcopy(
                self.ablation_models[seed].model
            ).to(self.device)
            ablation_record = self.checkpoint_metadata[
                "ablation_mean_signal"
            ][str(seed)]
            ablation_model.load_state_dict(torch.load(
                self.output_dir / ablation_record["path"],
                map_location=self.device,
                weights_only=True,
            ))
            systems = {
                "manual_score_h": test["heuristic_score"].to_numpy(float),
                "selected_ltr": predict_scores(selected_model, test),
                "ablation_mean_signal_ltr": predict_scores(ablation_model, test),
                "ablation_direct_probability": test["y_prob"].to_numpy(float),
            }
            for system, scores in systems.items():
                metrics = per_job_metrics(
                    test,
                    scores,
                    target_column=self.settings["gold"]["relevance_column"],
                    k_values=tuple(self.settings["gold"]["k_values"]),
                )
                metrics.insert(0, "seed", seed)
                metrics.insert(1, "system", system)
                per_job_rows.append(metrics)
                prediction_rows.append(pd.DataFrame({
                    "seed": seed,
                    "system": system,
                    "job_id": test["job_id"].to_numpy(),
                    "cand_id": test["cand_id"].to_numpy(),
                    "score": np.asarray(scores, dtype=float),
                }))
        self.gold_test_per_job = pd.concat(per_job_rows, ignore_index=True)
        averaged_per_job = self.gold_test_per_job.groupby(["system", "job_id"], as_index=False)[["ndcg@5", "ndcg@10", "mrr"]].mean()
        self.gold_test_summary = averaged_per_job.groupby("system", as_index=False)[["ndcg@5", "ndcg@10", "mrr"]].mean()
        comparisons = [
            ("manual_score_h", "selected_ltr", "main"),
            ("ablation_mean_signal_ltr", "selected_ltr", "core_1"),
            ("ablation_direct_probability", "selected_ltr", "core_2"),
        ]
        bootstrap_rows = []
        for baseline, proposed, comparison in comparisons:
            for metric in ["ndcg@5", "ndcg@10", "mrr"]:
                bootstrap_rows.append({
                    "comparison": comparison,
                    **paired_job_bootstrap(
                        averaged_per_job,
                        baseline,
                        proposed,
                        metric,
                        int(self.settings["bootstrap"]["n_resamples"]),
                        int(self.settings["bootstrap"]["seed"]),
                    ),
                })
        self.bootstrap_results = pd.DataFrame(bootstrap_rows)
        self.stage = "gold_test_evaluated"
        self.gold_test_per_job.to_csv(self.output_dir / "diagnostics" / "gold_test_per_job_by_seed.csv", index=False)
        self.gold_test_summary.to_csv(self.output_dir / "tables" / "gold_test_main_and_ablations.csv", index=False)
        self.bootstrap_results.to_csv(self.output_dir / "tables" / "paired_bootstrap_ci.csv", index=False)
        pd.concat(prediction_rows, ignore_index=True).to_csv(self.output_dir / "predictions" / "gold_test_predictions.csv", index=False)
        write_json(self.output_dir / "audit" / "protocol_final.json", self.protocol.manifest())
        return self.gold_test_summary.copy(), self.bootstrap_results.copy(), averaged_per_job

    def finalize(self) -> dict:
        if self.stage != "gold_test_evaluated":
            raise RuntimeError("Complete Gold-test evaluation before finalization")
        main = self.bootstrap_results[
            (self.bootstrap_results["comparison"] == "main")
            & (self.bootstrap_results["metric"] == "ndcg@5")
        ].iloc[0]
        conclusion = (
            "SUPPORTED: selected three-signal LTR improves nDCG@5 over the manual score."
            if bool(main["supports_improvement"])
            else "NOT SUPPORTED: the nDCG@5 confidence interval contains or touches zero."
        )
        manifest = {
            "protocol_version": self.settings["protocol_version"],
            "mode": "smoke" if self.smoke else "full",
            "stage": "complete",
            "selected_formulation": self.selected_formulation,
            "selected_posterior_threshold": self.posterior_threshold,
            "conclusion": conclusion,
            "counts": {
                "gold_validation_jobs": len(self.gold_manifest["validation_jobs"]),
                "gold_test_jobs": len(self.gold_manifest["test_jobs"]),
                "seeds": len(self.settings["seeds"]),
            },
            "input_files": build_input_manifest(self.data_root),
            "input_hashes": {
                "jobs": sha256_file(self.data_root / "JOB_DATA_FINAL.csv"),
                "candidates": sha256_file(self.data_root / "USER_DATA_FINAL.csv"),
                "gold": sha256_file(self.gold_path),
            },
        }
        write_json(self.output_dir / "audit" / "run_manifest.json", manifest)
        return manifest


def run_experiment(config_path: str | Path = "configs/experiment_3signal.yaml", smoke: bool = False) -> dict:
    experiment = ThreeSignalExperiment(config_path, smoke=smoke)
    experiment.audit_data_and_gold()
    experiment.prepare_development_data()
    experiment.fit_weak_supervision()
    experiment.train_and_select_formulation()
    experiment.evaluate_development_test_once()
    experiment.prepare_confirmatory_checkpoints()
    experiment.lock_protocol()
    experiment.evaluate_gold_test_once()
    return experiment.finalize()
