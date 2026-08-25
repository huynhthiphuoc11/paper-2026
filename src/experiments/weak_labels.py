from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import precision_score, recall_score

from src.experiments import LF_COLUMNS


_SIGNAL_TO_LF = {"s_sem": "lf_sem", "s_skill": "lf_skill", "s_exp": "lf_exp"}
_AVAILABILITY = {"s_sem": "sem_available", "s_skill": "skill_available", "s_exp": "exp_available"}


class PercentileLabelingFunctions:
    POLICY = "negative-global-positive-tail-v1"

    def __init__(
        self,
        negative_percentile: float = 25,
        positive_percentile: float = 75,
        threshold_policy: str = POLICY,
    ):
        if threshold_policy != self.POLICY:
            raise ValueError(f"Unsupported threshold policy: {threshold_policy}")
        self.negative_percentile = negative_percentile
        self.positive_percentile = positive_percentile
        self.threshold_policy = threshold_policy
        self.thresholds: dict[str, dict[str, float]] = {}
        self.threshold_diagnostics: dict[str, dict[str, int]] = {}
        self.is_fitted = False

    def fit(self, train: pd.DataFrame):
        thresholds = {}
        diagnostics = {}
        for signal in _SIGNAL_TO_LF:
            active = train.loc[train[_AVAILABILITY[signal]].astype(bool), signal].dropna().to_numpy(float)
            if not len(active):
                raise ValueError(f"No active train observations for {signal}")
            negative = float(np.percentile(active, self.negative_percentile))
            positive_tail = active[active > negative]
            if not len(positive_tail):
                raise ValueError(f"{signal} has no variation for labeling functions")
            positive = float(np.percentile(positive_tail, self.positive_percentile))
            thresholds[signal] = {"negative": negative, "positive": positive}
            diagnostics[signal] = {
                "n_active": int(len(active)),
                "n_at_or_below_negative": int((active <= negative).sum()),
                "n_positive_tail": int(len(positive_tail)),
            }
        self.thresholds = thresholds
        self.threshold_diagnostics = diagnostics
        self.is_fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Labeling functions must be fitted on train")
        output = frame.copy()
        for signal, lf_column in _SIGNAL_TO_LF.items():
            available = output[_AVAILABILITY[signal]].astype(bool) & output[signal].notna()
            values = output[signal].to_numpy(float)
            negative = self.thresholds[signal]["negative"]
            positive = self.thresholds[signal]["positive"]
            votes = np.zeros(len(output), dtype=int)
            votes[available.to_numpy() & (values <= negative)] = -1
            votes[available.to_numpy() & (values >= positive)] = 1
            output[lf_column] = votes
        return output


class DawidSkeneThreeSource:
    def __init__(
        self,
        n_iter: int = 100,
        parameter_clip: tuple[float, float] = (0.51, 0.99),
        prior_clip: tuple[float, float] = (0.05, 0.95),
        initial_accuracy: float = 0.75,
        convergence_tolerance: float = 1e-7,
    ):
        if int(n_iter) <= 0:
            raise ValueError("Dawid--Skene n_iter must be positive")
        if not 0.5 < parameter_clip[0] < parameter_clip[1] < 1.0:
            raise ValueError("parameter_clip must satisfy 0.5 < lower < upper < 1")
        if not 0.0 < prior_clip[0] < prior_clip[1] < 1.0:
            raise ValueError("prior_clip must satisfy 0 < lower < upper < 1")
        if not 0.5 < initial_accuracy < 1.0:
            raise ValueError("initial_accuracy must be in (0.5, 1)")
        if convergence_tolerance <= 0:
            raise ValueError("convergence_tolerance must be positive")
        self.n_iter = int(n_iter)
        self.parameter_clip = tuple(float(value) for value in parameter_clip)
        self.prior_clip = tuple(float(value) for value in prior_clip)
        self.initial_accuracy = float(initial_accuracy)
        self.convergence_tolerance = float(convergence_tolerance)
        self.prior = 0.5
        self.sensitivities: np.ndarray | None = None
        self.specificities: np.ndarray | None = None
        self.is_fitted = False

    @staticmethod
    def _posterior(matrix: np.ndarray, prior: float, sensitivity: np.ndarray, specificity: np.ndarray) -> np.ndarray:
        positive = matrix == 1
        negative = matrix == -1
        log_y1 = np.full(len(matrix), np.log(prior + 1e-12), dtype=float)
        log_y0 = np.full(len(matrix), np.log(1.0 - prior + 1e-12), dtype=float)
        for index in range(matrix.shape[1]):
            log_y1 += positive[:, index] * np.log(sensitivity[index] + 1e-12)
            log_y1 += negative[:, index] * np.log(1.0 - sensitivity[index] + 1e-12)
            log_y0 += positive[:, index] * np.log(1.0 - specificity[index] + 1e-12)
            log_y0 += negative[:, index] * np.log(specificity[index] + 1e-12)
        maximum = np.maximum(log_y1, log_y0)
        y1 = np.exp(log_y1 - maximum)
        y0 = np.exp(log_y0 - maximum)
        return y1 / (y1 + y0 + 1e-12)

    def fit(self, lf_frame: pd.DataFrame):
        matrix = lf_frame[LF_COLUMNS].to_numpy(int)
        if not len(matrix):
            raise ValueError("Cannot fit Dawid--Skene on empty data")
        active = matrix != 0
        active_rows = active.any(axis=1)
        if active_rows.any():
            positives = (matrix[active_rows] == 1).sum(axis=1)
            negatives = (matrix[active_rows] == -1).sum(axis=1)
            vote_direction = np.where(
                positives > negatives,
                1.0,
                np.where(negatives > positives, 0.0, np.nan),
            )
            prior = (
                float(np.nanmean(vote_direction))
                if np.isfinite(vote_direction).any()
                else 0.5
            )
        else:
            prior = 0.5
        prior = float(np.clip(prior, *self.prior_clip))
        sensitivity = np.full(
            matrix.shape[1], self.initial_accuracy, dtype=float
        )
        specificity = np.full(
            matrix.shape[1], self.initial_accuracy, dtype=float
        )
        previous = None
        for _ in range(self.n_iter):
            posterior = self._posterior(matrix, prior, sensitivity, specificity)
            prior = float(np.clip(posterior.mean(), 0.05, 0.95))
            for index in range(matrix.shape[1]):
                mask = active[:, index]
                positive_denom = posterior[mask].sum()
                negative_denom = (1.0 - posterior[mask]).sum()
                if positive_denom > 1e-12:
                    sensitivity[index] = np.clip(
                        (posterior * (matrix[:, index] == 1)).sum() / positive_denom,
                        *self.parameter_clip,
                    )
                if negative_denom > 1e-12:
                    specificity[index] = np.clip(
                        ((1.0 - posterior) * (matrix[:, index] == -1)).sum() / negative_denom,
                        *self.parameter_clip,
                    )
            if (
                previous is not None
                and np.max(np.abs(posterior - previous))
                < self.convergence_tolerance
            ):
                break
            previous = posterior
        self.prior = prior
        self.sensitivities = sensitivity
        self.specificities = specificity
        self.is_fitted = True
        return self

    def predict_proba(self, lf_frame: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Dawid--Skene must be fitted on train")
        return self._posterior(
            lf_frame[LF_COLUMNS].to_numpy(int),
            self.prior,
            self.sensitivities,
            self.specificities,
        )

    def parameters(self) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Dawid--Skene is not fitted")
        return {
            "prior": float(self.prior),
            "lf_columns": list(LF_COLUMNS),
            "sensitivities": self.sensitivities.tolist(),
            "specificities": self.specificities.tolist(),
            "assumptions": {
                "n_iter": self.n_iter,
                "parameter_clip": list(self.parameter_clip),
                "prior_clip": list(self.prior_clip),
                "initial_accuracy": self.initial_accuracy,
                "convergence_tolerance": self.convergence_tolerance,
            },
        }


class ThreeSourceWeakLabelPipeline:
    def __init__(
        self,
        negative_percentile: float = 25,
        positive_percentile: float = 75,
        threshold_policy: str = PercentileLabelingFunctions.POLICY,
        label_model_config: dict | None = None,
    ):
        self.labeling_functions = PercentileLabelingFunctions(
            negative_percentile, positive_percentile, threshold_policy
        )
        self.label_model = DawidSkeneThreeSource(**(label_model_config or {}))
        self.is_fitted = False

    def fit_transform_train(self, train: pd.DataFrame) -> pd.DataFrame:
        lfs = self.labeling_functions.fit(train).transform(train)
        self.label_model.fit(lfs)
        self.is_fitted = True
        output = lfs.copy()
        output["y_prob"] = self.label_model.predict_proba(lfs)
        return output

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Weak-label pipeline must be fitted on train")
        lfs = self.labeling_functions.transform(frame)
        output = lfs.copy()
        output["y_prob"] = self.label_model.predict_proba(lfs)
        return output

    def parameters(self) -> dict:
        return {
            "threshold_policy": self.labeling_functions.threshold_policy,
            "negative_percentile": self.labeling_functions.negative_percentile,
            "positive_percentile": self.labeling_functions.positive_percentile,
            "thresholds": self.labeling_functions.thresholds,
            "threshold_diagnostics": self.labeling_functions.threshold_diagnostics,
            "label_model": self.label_model.parameters(),
        }


def strict_three_of_three(lf_frame: pd.DataFrame) -> np.ndarray:
    matrix = lf_frame[LF_COLUMNS].to_numpy(int)
    prediction = np.full(len(matrix), np.nan)
    prediction[(matrix == 1).all(axis=1)] = 1.0
    prediction[(matrix == -1).all(axis=1)] = 0.0
    return prediction


def lf_diagnostics(lf_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = []
    for column in LF_COLUMNS:
        values = lf_frame[column].to_numpy(int)
        stats.append({
            "lf": column,
            "n_total": int(len(values)),
            "n_positive": int((values == 1).sum()),
            "n_negative": int((values == -1).sum()),
            "n_abstain": int((values == 0).sum()),
            "coverage": float(np.mean(values != 0)),
            "positive_rate": float(np.mean(values == 1)),
            "negative_rate": float(np.mean(values == -1)),
            "abstain_rate": float(np.mean(values == 0)),
        })
    pair_rows = []
    for left, right in itertools.combinations(LF_COLUMNS, 2):
        left_values = lf_frame[left].to_numpy(int)
        right_values = lf_frame[right].to_numpy(int)
        joint = (left_values != 0) & (right_values != 0)
        if joint.any():
            joint_left = left_values[joint]
            joint_right = right_values[joint]
            if len(np.unique(joint_left)) < 2 or len(np.unique(joint_right)) < 2:
                spearman = 0.0
            else:
                rho, _ = spearmanr(joint_left, joint_right)
                spearman = float(np.nan_to_num(rho))
            agreement = float(np.mean(joint_left == joint_right))
            conflict = float(np.mean(joint_left != joint_right))
        else:
            agreement = np.nan
            conflict = np.nan
            spearman = np.nan
        pair_rows.append({
            "lf_a": left,
            "lf_b": right,
            "joint_coverage": float(joint.mean()),
            "agreement": agreement,
            "conflict": conflict,
            "spearman": spearman,
        })
    return pd.DataFrame(stats), pd.DataFrame(pair_rows)


def binary_quality(y_true: np.ndarray, predictions: np.ndarray) -> dict:
    observed = np.isfinite(predictions)
    truth_all = np.asarray(y_true, dtype=int)
    if not observed.any():
        return {
            "precision": np.nan,
            "recall": np.nan,
            "coverage": 0.0,
            "n_covered": 0,
            "n_positive": int(truth_all.sum()),
            "n_predicted_positive": 0,
            "tp": 0,
            "fp": 0,
            "fn": int(truth_all.sum()),
            "tn": int((truth_all == 0).sum()),
        }
    truth = truth_all[observed]
    predicted = np.asarray(predictions, dtype=float)[observed].astype(int)
    tp = int(((truth == 1) & (predicted == 1)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    tn = int(((truth == 0) & (predicted == 0)).sum())
    return {
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "coverage": float(observed.mean()),
        "n_covered": int(observed.sum()),
        "n_positive": int(truth.sum()),
        "n_predicted_positive": int(predicted.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

