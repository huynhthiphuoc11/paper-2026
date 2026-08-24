from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from sklearn.metrics import log_loss

from src.data.feature_pipeline import (
    TrainOnlyFeaturePipeline,
    load_gold_with_identity_check,
    sample_raw_entities,
    split_pairs_by_job,
)
from src.data.loader import FEATURE_COLS
from src.eval.metrics import paired_job_bootstrap, per_job_ranking_metrics
from src.eval.perturbation import apply_isolated_perturbation
from src.models.pairing import build_pair_table, pair_table_hash
from src.models.training import (
    FixedHeuristic,
    PointwiseLogistic,
    predict_linear_soft_bce,
    predict_ranknet,
    train_linear_soft_bce,
    train_ranknet,
)
from src.weak.pipeline import LF_COLS, WeakLabelPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run the authoritative paper experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def resolved_settings(config: dict, smoke: bool) -> dict:
    settings = json.loads(json.dumps(config))
    if smoke:
        smoke_config = config["smoke"]
        settings["seeds"] = smoke_config["seeds"]
        settings["models"]["B1"]["c_grid"] = smoke_config["B1_c_grid"]
        settings["models"]["B2"]["learning_rate_grid"] = smoke_config[
            "B2_learning_rate_grid"
        ]
        settings["models"]["B2"]["batch_size_grid"] = smoke_config[
            "B2_batch_size_grid"
        ]
        settings["models"]["B2"]["max_epochs"] = smoke_config[
            "B2_max_epochs"
        ]
        settings["models"]["B2"]["patience"] = smoke_config["patience"]
        settings["models"]["ranknet"]["learning_rate_grid"] = smoke_config[
            "ranknet_learning_rate_grid"
        ]
        settings["models"]["ranknet"]["batch_size_grid"] = smoke_config[
            "ranknet_batch_size_grid"
        ]
        settings["models"]["ranknet"]["max_epochs"] = smoke_config[
            "ranknet_max_epochs"
        ]
        settings["models"]["ranknet"]["patience"] = smoke_config["patience"]
        settings["models"]["M2"]["lambda_grid"] = smoke_config["lambda_grid"]
        settings["bootstrap"]["n_resamples"] = smoke_config[
            "bootstrap_resamples"
        ]
    settings["run_mode"] = "smoke" if smoke else "full"
    return settings


def prepare_output_tree(output_root: Path):
    for directory in [
        "predictions",
        "metrics",
        "diagnostics",
        "tables",
        "audit",
        "checkpoints",
    ]:
        (output_root / directory).mkdir(parents=True, exist_ok=True)


def validate_config(config: dict):
    if config["features"]["columns"] != FEATURE_COLS:
        raise ValueError(f"Feature contract must be exactly {FEATURE_COLS}")
    if config["models"]["ranknet"]["architecture"] != [5, 32, 16, 1]:
        raise ValueError("RankNet architecture must be exactly 5 -> 32 -> 16 -> 1")
    if config["gold"]["binary_relevance_threshold"] != 2:
        raise ValueError("MAP protocol requires relevance >= 2")
    if config["pairing"]["pair_delta"] < 0:
        raise ValueError("pair_delta must be non-negative")


def lf_statistics(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for column in LF_COLS:
        values = frame[column].to_numpy(int)
        rows.append(
            {
                "seed": seed,
                "lf_name": column,
                "n": len(values),
                "coverage_rate": float(np.mean(values != 0)),
                "positive_rate": float(np.mean(values == 1)),
                "negative_rate": float(np.mean(values == -1)),
                "abstain_rate": float(np.mean(values == 0)),
            }
        )
    return pd.DataFrame(rows)


def lf_agreement(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for left, right in itertools.combinations(LF_COLS, 2):
        left_values = frame[left].to_numpy(int)
        right_values = frame[right].to_numpy(int)
        jointly_active = (left_values != 0) & (right_values != 0)
        conflict = jointly_active & (left_values != right_values)
        agreement = jointly_active & (left_values == right_values)
        correlation = pd.Series(left_values).corr(
            pd.Series(right_values), method="spearman"
        )
        rows.append(
            {
                "seed": seed,
                "lf_a": left,
                "lf_b": right,
                "joint_coverage_rate": float(jointly_active.mean()),
                "agreement_rate": float(
                    agreement.sum() / jointly_active.sum()
                )
                if jointly_active.any()
                else np.nan,
                "conflict_rate": float(conflict.mean()),
                "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def ds_statistics(frame: pd.DataFrame, parameters: dict, seed: int) -> pd.DataFrame:
    probabilities = frame["y_prob"].to_numpy(float)
    parameter_values = np.asarray(
        parameters["source_sensitivities"] + parameters["source_specificities"],
        dtype=float,
    )
    return pd.DataFrame(
        [
            {
                "seed": seed,
                "n": len(frame),
                "y_prob_mean": float(probabilities.mean()),
                "y_prob_std": float(probabilities.std()),
                "high_confidence_positive": int((probabilities >= 0.8).sum()),
                "high_confidence_negative": int((probabilities <= 0.2).sum()),
                "ambiguous": int(
                    ((probabilities > 0.4) & (probabilities < 0.6)).sum()
                ),
                "p_prior": parameters["p_prior"],
                "prior_at_boundary": bool(
                    np.isclose(parameters["p_prior"], 0.05)
                    or np.isclose(parameters["p_prior"], 0.95)
                ),
                "parameter_boundary_rate": float(
                    np.mean(
                        np.isclose(parameter_values, 0.55)
                        | np.isclose(parameter_values, 0.95)
                    )
                ),
            }
        ]
    )


def select_b1(train, validation, c_grid, seed):
    best = None
    records = []
    target = validation["heuristic_label"].to_numpy(int)
    for c_value in c_grid:
        model = PointwiseLogistic(c=float(c_value), seed=seed).fit(train)
        probability = np.clip(model.predict(validation), 1e-7, 1 - 1e-7)
        loss = float(log_loss(target, probability, labels=[0, 1]))
        records.append({"C": float(c_value), "validation_log_loss": loss})
        if best is None or loss < best[0]:
            best = (loss, model, float(c_value))
    return best[1], {"selected_C": best[2], "validation_log_loss": best[0], "grid": records}


def select_b2(train, validation, model_config, seed):
    best = None
    records = []
    for learning_rate, batch_size in itertools.product(
        model_config["learning_rate_grid"], model_config["batch_size_grid"]
    ):
        result = train_linear_soft_bce(
            train,
            validation,
            seed=seed,
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
            max_epochs=int(model_config["max_epochs"]),
            patience=int(model_config["patience"]),
        )
        record = result.metadata()
        records.append(record)
        if best is None or result.best_validation_loss < best.best_validation_loss:
            best = result
    return best, records


def select_m1(
    train,
    validation,
    train_pairs,
    validation_pairs,
    model_config,
    seed,
):
    best = None
    records = []
    for learning_rate, batch_size in itertools.product(
        model_config["learning_rate_grid"], model_config["batch_size_grid"]
    ):
        result = train_ranknet(
            train,
            validation,
            train_pairs,
            validation_pairs,
            seed=seed,
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
            max_epochs=int(model_config["max_epochs"]),
            patience=int(model_config["patience"]),
            lambda_gap=0.0,
            dropout=float(model_config["dropout"]),
        )
        records.append(result.metadata())
        if (
            best is None
            or result.best_validation_rank_loss < best.best_validation_rank_loss
        ):
            best = result
    return best, records


def select_m2(
    train,
    validation,
    train_pairs,
    validation_pairs,
    ranknet_config,
    lambda_grid,
    selected_learning_rate,
    selected_batch_size,
    seed,
):
    best = None
    records = []
    for lambda_gap in lambda_grid:
        result = train_ranknet(
            train,
            validation,
            train_pairs,
            validation_pairs,
            seed=seed,
            learning_rate=float(selected_learning_rate),
            batch_size=int(selected_batch_size),
            max_epochs=int(ranknet_config["max_epochs"]),
            patience=int(ranknet_config["patience"]),
            lambda_gap=float(lambda_gap),
            dropout=float(ranknet_config["dropout"]),
        )
        records.append(result.metadata())
        if (
            best is None
            or result.best_validation_rank_loss < best.best_validation_rank_loss
        ):
            best = result
    return best, records


def prediction_frame(gold, scores, model, seed):
    return pd.DataFrame(
        {
            "seed": seed,
            "job_id": gold["job_id"].to_numpy(),
            "cv_id": gold["cand_id"].to_numpy(),
            "score": np.asarray(scores, dtype=float),
            "model": model,
        }
    )


def perturbation_rows(gold, predict_fn, model: str, seed: int, config: dict):
    base_scores = predict_fn(gold)
    rows = []
    for kind in config["kinds"]:
        perturbed = pd.DataFrame(
            [
                apply_isolated_perturbation(
                    row,
                    kind,
                    skill_drop=float(config["skill_drop"]),
                    exp_drop=float(config["exp_drop"]),
                    domain_role_cap=float(config["domain_role_cap"]),
                )
                for _, row in gold.iterrows()
            ]
        )
        perturbed_scores = predict_fn(perturbed)
        held = (base_scores > perturbed_scores).astype(float)
        for (job_id, group_indices) in gold.groupby("job_id", sort=True).groups.items():
            indices = np.asarray(list(group_indices), dtype=int)
            rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "perturbation": kind,
                    "job_id": job_id,
                    "qual_sens": float(held[indices].mean()),
                    "n_pairs": len(indices),
                }
            )
    return rows


def average_seed_metrics(per_seed_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in per_seed_metrics if "@" in column]
    return (
        per_seed_metrics.groupby(["model", "job_id"], as_index=False)[metric_columns]
        .mean()
        .sort_values(["model", "job_id"])
    )


def build_bootstrap_results(per_job_average, config):
    rows = []
    comparisons = [("H", "M1"), ("B1", "M1"), ("B2", "M1")]
    metric_columns = [column for column in per_job_average if "@" in column]
    for model_a, model_b in comparisons:
        for metric in metric_columns:
            rows.append(
                paired_job_bootstrap(
                    per_job_average,
                    model_a,
                    model_b,
                    metric,
                    n_bootstraps=int(config["n_resamples"]),
                    seed=int(config["seed"]),
                )
            )
    return pd.DataFrame(rows)


def build_perturbation_bootstrap(per_job_perturbation, config):
    averaged = (
        per_job_perturbation.groupby(
            ["model", "perturbation", "job_id"], as_index=False
        )["qual_sens"]
        .mean()
    )
    rows = []
    for kind, group in averaged.groupby("perturbation"):
        result = paired_job_bootstrap(
            group,
            "M1",
            "M2",
            "qual_sens",
            n_bootstraps=int(config["n_resamples"]),
            seed=int(config["seed"]),
        )
        result["perturbation"] = kind
        rows.append(result)
    return averaged, pd.DataFrame(rows)


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = resolved_settings(yaml.safe_load(handle), args.smoke)
    validate_config(config)

    output_root = resolve_path(config["output_dir"])
    data_dir = resolve_path(config["data_dir"])
    gold_path = resolve_path(config["gold"]["path"])
    prepare_output_tree(output_root)
    write_json(output_root / "audit" / "resolved_config.json", config)

    gold = load_gold_with_identity_check(str(data_dir), str(gold_path))
    gold_jobs = set(gold["job_id"])
    gold_candidates = set(gold["cand_id"])

    sampled = sample_raw_entities(
        str(data_dir),
        n_jobs=int(config["sample"]["n_jobs"]),
        n_candidates=int(config["sample"]["n_candidates"]),
        candidates_per_job=int(config["sample"]["candidates_per_job"]),
        seed=int(config["seed"]),
        excluded_job_ids=gold_jobs,
        excluded_candidate_ids=gold_candidates,
    )
    if len(sampled.jobs) != int(config["sample"]["n_jobs"]):
        raise ValueError("Unable to sample the configured number of non-gold jobs")
    if set(sampled.pairs["job_id"]) & gold_jobs:
        raise AssertionError("Gold job leaked into weak-label pool")
    if set(sampled.pairs["cand_id"]) & gold_candidates:
        raise AssertionError("Gold candidate leaked into weak-label pool")

    train_pairs, validation_pairs, test_pairs, split_manifest = split_pairs_by_job(
        sampled.pairs,
        seed=int(config["split"]["seed"]),
        train_ratio=float(config["split"]["train_ratio"]),
        val_ratio=float(config["split"]["validation_ratio"]),
    )
    all_split_jobs = (
        set(split_manifest["train_jobs"])
        | set(split_manifest["validation_jobs"])
        | set(split_manifest["test_jobs"])
    )
    if all_split_jobs & gold_jobs:
        raise AssertionError("Gold jobs must be disjoint from all development splits")

    feature_pipeline = TrainOnlyFeaturePipeline(
        df_threshold=config["features"]["high_df_skill_threshold"],
        role_max_features=int(config["features"]["role_tfidf_max_features"]),
        description_max_features=int(
            config["features"]["description_tfidf_max_features"]
        ),
    ).fit(train_pairs, sampled.jobs, sampled.candidates)
    train = feature_pipeline.transform(
        train_pairs, sampled.jobs, sampled.candidates
    ).reset_index(drop=True)
    validation = feature_pipeline.transform(
        validation_pairs, sampled.jobs, sampled.candidates
    ).reset_index(drop=True)
    test = feature_pipeline.transform(
        test_pairs, sampled.jobs, sampled.candidates
    ).reset_index(drop=True)
    if set(feature_pipeline.fit_job_ids) & gold_jobs:
        raise AssertionError("Gold job was used to fit preprocessing")
    if set(feature_pipeline.fit_candidate_ids) & gold_candidates:
        raise AssertionError("Gold candidate was used to fit preprocessing")

    weak_pipeline = WeakLabelPipeline(
        pos_percentile=float(config["weak_supervision"]["pos_percentile"]),
        neg_percentile=float(config["weak_supervision"]["neg_percentile"]),
    )
    method = config["weak_supervision"]["aggregator"]
    train_weak = weak_pipeline.fit_transform_train(train, method=method)
    parameters_before = weak_pipeline.aggregator_parameters()
    validation_weak = weak_pipeline.transform(validation, method=method)
    test_weak = weak_pipeline.transform(test, method=method)
    if parameters_before != weak_pipeline.aggregator_parameters():
        raise AssertionError("Validation/test inference mutated weak-label model")

    weak_columns = [
        "pair_id",
        "job_id",
        "cand_id",
        *LF_COLS,
        "y_prob",
        "y_weak_consensus",
    ]
    train_weak[weak_columns].to_csv(
        output_root / "audit" / "train_weak_labels.csv", index=False
    )
    write_json(output_root / "audit" / "preprocessing_manifest.json", feature_pipeline.manifest())
    write_json(output_root / "audit" / "split_manifest.json", split_manifest)
    write_json(output_root / "audit" / "label_model_parameters.json", parameters_before)

    prediction_rows = {model: [] for model in ["H", "B1", "B2", "M1", "M2"]}
    metric_rows = []
    perturbation_metric_rows = []
    selection_records = []
    pair_diagnostic_rows = []
    lf_stat_rows = []
    lf_agreement_rows = []
    ds_stat_rows = []
    pair_hash_rows = []

    for seed in config["seeds"]:
        seed = int(seed)
        print(f"Running seed {seed} ({config['run_mode']})", flush=True)
        lf_stat_rows.append(lf_statistics(train_weak, seed))
        lf_agreement_rows.append(lf_agreement(train_weak, seed))
        ds_stat_rows.append(ds_statistics(train_weak, parameters_before, seed))

        train_pair_table, train_pair_diagnostics = build_pair_table(
            train_weak,
            score_col=config["pairing"]["score_column"],
            pair_delta=float(config["pairing"]["pair_delta"]),
            max_pairs_per_job=int(config["pairing"]["max_pairs_per_job"]),
            seed=seed,
        )
        validation_pair_table, validation_pair_diagnostics = build_pair_table(
            validation_weak,
            score_col=config["pairing"]["score_column"],
            pair_delta=float(config["pairing"]["pair_delta"]),
            max_pairs_per_job=int(config["pairing"]["max_pairs_per_job"]),
            seed=seed + 1,
        )
        if train_pair_table.empty or validation_pair_table.empty:
            raise RuntimeError(
                "Fixed pair_delta yielded an empty train/validation pair table"
            )
        train_hash = pair_table_hash(train_pair_table)
        validation_hash = pair_table_hash(validation_pair_table)
        pair_hash_rows.append(
            {
                "seed": seed,
                "m1_train_pair_hash": train_hash,
                "m2_train_pair_hash": train_hash,
                "m1_validation_pair_hash": validation_hash,
                "m2_validation_pair_hash": validation_hash,
            }
        )
        for split, diagnostics in [
            ("train", train_pair_diagnostics),
            ("validation", validation_pair_diagnostics),
        ]:
            diagnostics = diagnostics.copy()
            diagnostics.insert(0, "seed", seed)
            diagnostics.insert(1, "split", split)
            pair_diagnostic_rows.append(diagnostics)

        heuristic = FixedHeuristic().fit(train_weak)
        b1, b1_selection = select_b1(
            train_weak,
            validation_weak,
            config["models"]["B1"]["c_grid"],
            seed,
        )
        b2, b2_grid = select_b2(
            train_weak, validation_weak, config["models"]["B2"], seed
        )
        m1, m1_grid = select_m1(
            train_weak,
            validation_weak,
            train_pair_table,
            validation_pair_table,
            config["models"]["ranknet"],
            seed,
        )
        m2, m2_grid = select_m2(
            train_weak,
            validation_weak,
            train_pair_table,
            validation_pair_table,
            config["models"]["ranknet"],
            config["models"]["M2"]["lambda_grid"],
            m1.learning_rate,
            m1.batch_size,
            seed,
        )

        feature_state_before_gold = feature_pipeline.manifest()
        label_state_before_gold = weak_pipeline.aggregator_parameters()
        gold_features = feature_pipeline.transform_gold(
            gold, str(data_dir)
        ).reset_index(drop=True)
        if not gold_features[config["gold"]["relevance_column"]].isin(
            config["gold"]["grade_scale"]
        ).all():
            raise ValueError("Graded relevance is outside the locked 0-3 scale")
        gold_weak = weak_pipeline.transform(gold_features, method=method)
        if feature_pipeline.manifest() != feature_state_before_gold:
            raise AssertionError("Gold transform mutated fitted preprocessing state")
        if weak_pipeline.aggregator_parameters() != label_state_before_gold:
            raise AssertionError("Gold inference mutated weak-label model")
        gold_weak[config["gold"]["relevance_column"]] = gold_features[
            config["gold"]["relevance_column"]
        ].to_numpy()

        predictors = {
            "H": heuristic.predict,
            "B1": b1.predict,
            "B2": lambda frame, model=b2.model: predict_linear_soft_bce(model, frame),
            "M1": lambda frame, model=m1.model: predict_ranknet(model, frame),
            "M2": lambda frame, model=m2.model: predict_ranknet(model, frame),
        }
        selection_records.append(
            {
                "seed": seed,
                "B1": b1_selection,
                "B1_parameters": b1.parameters(),
                "B2_grid": b2_grid,
                "B2_selected": b2.metadata(),
                "M1_grid": m1_grid,
                "M1_selected": m1.metadata(),
                "M2_grid": m2_grid,
                "M2_selected": m2.metadata(),
            }
        )
        torch.save(b2.model.state_dict(), output_root / "checkpoints" / f"B2_seed{seed}.pt")
        torch.save(m1.model.state_dict(), output_root / "checkpoints" / f"M1_seed{seed}.pt")
        torch.save(m2.model.state_dict(), output_root / "checkpoints" / f"M2_seed{seed}.pt")
        torch.save(m2.gap_head.state_dict(), output_root / "checkpoints" / f"M2_gap_seed{seed}.pt")

        for model_name, predict_fn in predictors.items():
            scores = predict_fn(gold_weak)
            predictions = prediction_frame(
                gold_weak, scores, model_name, seed
            )
            prediction_rows[model_name].append(predictions)
            metrics = per_job_ranking_metrics(
                gold_weak,
                scores,
                target_col=config["gold"]["relevance_column"],
                k_list=tuple(config["gold"]["k_values"]),
            )
            metrics.insert(0, "seed", seed)
            metrics.insert(1, "model", model_name)
            metric_rows.append(metrics)
            if model_name in {"M1", "M2"}:
                perturbation_metric_rows.extend(
                    perturbation_rows(
                        gold_weak,
                        predict_fn,
                        model_name,
                        seed,
                        config["perturbation"],
                    )
                )

    for model_name, frames in prediction_rows.items():
        output = pd.concat(frames, ignore_index=True)
        output[["seed", "job_id", "cv_id", "score"]].to_csv(
            output_root / "predictions" / f"{model_name}.csv", index=False
        )

    per_seed_metrics = pd.concat(metric_rows, ignore_index=True)
    averaged_metrics = average_seed_metrics(per_seed_metrics)
    bootstrap_results = build_bootstrap_results(
        averaged_metrics, config["bootstrap"]
    )
    perturbation_per_job = pd.DataFrame(perturbation_metric_rows)
    perturbation_average, perturbation_bootstrap = build_perturbation_bootstrap(
        perturbation_per_job, config["bootstrap"]
    )

    per_seed_metrics.to_csv(
        output_root / "metrics" / "ranking_metrics.csv", index=False
    )
    bootstrap_results.to_csv(
        output_root / "metrics" / "bootstrap_ci.csv", index=False
    )
    perturbation_per_job.to_csv(
        output_root / "metrics" / "perturbation_metrics.csv", index=False
    )
    pd.concat(lf_stat_rows, ignore_index=True).to_csv(
        output_root / "diagnostics" / "lf_statistics.csv", index=False
    )
    pd.concat(lf_agreement_rows, ignore_index=True).to_csv(
        output_root / "diagnostics" / "lf_agreement.csv", index=False
    )
    pd.concat(ds_stat_rows, ignore_index=True).to_csv(
        output_root / "diagnostics" / "ds_statistics.csv", index=False
    )
    pd.concat(pair_diagnostic_rows, ignore_index=True).to_csv(
        output_root / "diagnostics" / "pair_sampling.csv", index=False
    )

    metric_columns = [column for column in averaged_metrics if "@" in column]
    main_results = (
        averaged_metrics[averaged_metrics["model"].isin(["H", "B1", "B2", "M1"])]
        .groupby("model", as_index=False)[metric_columns]
        .mean()
    )
    main_results.to_csv(output_root / "tables" / "main_results.csv", index=False)
    bootstrap_results.to_csv(
        output_root / "tables" / "bootstrap_results.csv", index=False
    )
    perturbation_table = (
        perturbation_average.groupby(["model", "perturbation"], as_index=False)[
            "qual_sens"
        ]
        .mean()
        .merge(
            perturbation_bootstrap[
                [
                    "perturbation",
                    "mean_delta",
                    "ci_95_low",
                    "ci_95_high",
                    "n_jobs",
                ]
            ],
            on="perturbation",
            how="left",
        )
    )
    perturbation_table.to_csv(
        output_root / "tables" / "perturbation_results.csv", index=False
    )

    write_json(output_root / "audit" / "model_selection.json", selection_records)
    pd.DataFrame(pair_hash_rows).to_csv(
        output_root / "audit" / "pair_hashes.csv", index=False
    )
    write_json(
        output_root / "audit" / "run_manifest.json",
        {
            "protocol_version": config["protocol_version"],
            "run_mode": config["run_mode"],
            "dataset_files": {
                "jobs": sha256_file(data_dir / "JOB_DATA_FINAL.csv"),
                "candidates": sha256_file(data_dir / "USER_DATA_FINAL.csv"),
                "graded_evaluation": sha256_file(gold_path),
            },
            "source_files": {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in [
                    ROOT / "src" / "data" / "feature_pipeline.py",
                    ROOT / "src" / "weak" / "aggregator.py",
                    ROOT / "src" / "weak" / "lf.py",
                    ROOT / "src" / "models" / "pairing.py",
                    ROOT / "src" / "models" / "training.py",
                    ROOT / "src" / "eval" / "metrics.py",
                    Path(__file__).resolve(),
                ]
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
            },
            "counts": {
                "sampled_jobs": len(sampled.jobs),
                "sampled_candidates": len(sampled.candidates),
                "sampled_pairs": len(sampled.pairs),
                "train_pairs": len(train),
                "validation_pairs": len(validation),
                "test_pairs": len(test),
                "graded_pairs": len(gold),
                "graded_jobs": gold["job_id"].nunique(),
            },
        },
    )
    print(f"Completed authoritative run: {output_root}", flush=True)


if __name__ == "__main__":
    main()
