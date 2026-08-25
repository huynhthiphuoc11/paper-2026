from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.experiments import FEATURE_COLUMNS
from src.experiments.models import LinearScorer
from src.experiments.utils import set_seed


@dataclass
class PreparedDataset:
    features: torch.Tensor
    targets: torch.Tensor
    query_order: tuple[object, ...]
    query_indices: dict[object, torch.Tensor]


@dataclass
class PairwiseState:
    train_table: pd.DataFrame
    validation_table: pd.DataFrame
    train_tensors: tuple[torch.Tensor, torch.Tensor]
    validation_tensors: tuple[torch.Tensor, torch.Tensor]
    train_hash: str
    validation_hash: str


@dataclass
class TrainingResult:
    formulation: str
    model: LinearScorer
    learning_rate: float
    batch_size: int
    best_epoch: int
    best_validation_loss: float
    best_train_loss: float
    epochs_ran: int
    stopped_early: bool
    weight_decay: float
    gradient_clip_norm: float
    selection_metric: str
    best_validation_ndcg_at_5: float
    batch_unit: str
    device: str
    history: pd.DataFrame
    train_pair_hash: str | None = None
    validation_pair_hash: str | None = None

    def metadata(self) -> dict:
        return {
            "formulation": self.formulation,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "best_train_loss": self.best_train_loss,
            "generalization_gap": self.best_validation_loss - self.best_train_loss,
            "epochs_ran": self.epochs_ran,
            "stopped_early": self.stopped_early,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "selection_metric": self.selection_metric,
            "best_validation_ndcg_at_5": self.best_validation_ndcg_at_5,
            "batch_unit": self.batch_unit,
            "device": self.device,
            "train_pair_hash": self.train_pair_hash,
            "validation_pair_hash": self.validation_pair_hash,
        }


def _resolve_device(device: str | torch.device = "cpu") -> torch.device:
    requested = torch.device(device)
    if requested.type == "cpu":
        return requested
    if requested.type != "cuda":
        raise ValueError(f"Unsupported training device: {requested}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available in this PyTorch environment"
        )
    index = 0 if requested.index is None else requested.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {index} is outside the available device range"
        )
    return torch.device(f"cuda:{index}")


def _model_device(model: LinearScorer) -> torch.device:
    return next(model.parameters()).device


def _tensor_features(frame: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(frame[FEATURE_COLUMNS].to_numpy(float), dtype=torch.float32)


def _materialize_dataset(
    frame: pd.DataFrame,
    device: str | torch.device = "cpu",
) -> PreparedDataset:
    resolved = _resolve_device(device)
    features = _tensor_features(frame).to(resolved)
    targets = torch.tensor(
        frame["y_prob"].to_numpy(float), dtype=torch.float32, device=resolved
    )
    job_ids = frame["job_id"].to_numpy()
    query_order = tuple(sorted(pd.unique(job_ids).tolist()))
    query_indices = {
        job_id: torch.tensor(
            np.flatnonzero(job_ids == job_id), dtype=torch.long, device=resolved
        )
        for job_id in query_order
    }
    return PreparedDataset(features, targets, query_order, query_indices)


def predict_scores(
    model: LinearScorer,
    data: pd.DataFrame | PreparedDataset,
) -> np.ndarray:
    model.eval()
    device = _model_device(model)
    features = (
        data.features.to(device)
        if isinstance(data, PreparedDataset)
        else _tensor_features(data).to(device)
    )
    with torch.no_grad():
        return model(features).detach().cpu().numpy()


def _batches(size: int, batch_size: int, rng: np.random.RandomState):
    indices = np.arange(size)
    rng.shuffle(indices)
    for start in range(0, size, batch_size):
        yield indices[start:start + batch_size]


def build_pair_table(
    frame: pd.DataFrame,
    pair_delta: float,
    max_pairs_per_job: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for job_id, group in frame.groupby("job_id", sort=True):
        group = group.sort_values(["cand_id", "pair_id"], kind="mergesort")
        indices = group.index.to_numpy(int)
        probabilities = group["y_prob"].to_numpy(float)
        pair_ids = group["pair_id"].to_numpy(int)
        left_positions, right_positions = np.triu_indices(len(group), k=1)
        differences = probabilities[left_positions] - probabilities[right_positions]
        eligible_mask = (~np.isclose(differences, 0.0)) & (np.abs(differences) >= pair_delta)
        left_positions = left_positions[eligible_mask]
        right_positions = right_positions[eligible_mask]
        differences = differences[eligible_mask]
        if len(differences) > max_pairs_per_job:
            selected = rng.choice(
                len(differences), size=max_pairs_per_job, replace=False
            )
            left_positions = left_positions[selected]
            right_positions = right_positions[selected]
            differences = differences[selected]
        positive = differences > 0
        preferred_positions = np.where(positive, left_positions, right_positions)
        nonpreferred_positions = np.where(positive, right_positions, left_positions)
        eligible = [
            {
                "job_id": job_id,
                "preferred_index": int(indices[preferred]),
                "nonpreferred_index": int(indices[nonpreferred]),
                "preferred_pair_id": int(pair_ids[preferred]),
                "nonpreferred_pair_id": int(pair_ids[nonpreferred]),
                "delta": float(abs(difference)),
            }
            for preferred, nonpreferred, difference in zip(
                preferred_positions, nonpreferred_positions, differences
            )
        ]
        rows.extend(sorted(
            eligible,
            key=lambda row: (
                row["job_id"], row["preferred_pair_id"],
                row["nonpreferred_pair_id"],
            ),
        ))
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(f"No RankNet pairs at fixed delta {pair_delta}; no fallback is permitted")
    if (table["delta"] < pair_delta).any():
        raise AssertionError("Pair below fixed delta")
    return table.reset_index(drop=True)


def pair_table_hash(table: pd.DataFrame) -> str:
    columns = ["job_id", "preferred_pair_id", "nonpreferred_pair_id"]
    payload = table[columns].sort_values(columns, kind="mergesort").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pair_tensors(
    frame: pd.DataFrame,
    table: pd.DataFrame,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved = _resolve_device(device)
    preferred = frame.loc[table["preferred_index"].astype(int), FEATURE_COLUMNS].to_numpy(float)
    nonpreferred = frame.loc[table["nonpreferred_index"].astype(int), FEATURE_COLUMNS].to_numpy(float)
    return (
        torch.tensor(preferred, dtype=torch.float32, device=resolved),
        torch.tensor(nonpreferred, dtype=torch.float32, device=resolved),
    )


def pointwise_loss(model: LinearScorer, frame: pd.DataFrame) -> torch.Tensor:
    device = _model_device(model)
    target = torch.tensor(
        frame["y_prob"].to_numpy(float), dtype=torch.float32, device=device
    )
    return F.binary_cross_entropy_with_logits(
        model(_tensor_features(frame).to(device)), target
    )


def _pointwise_tensor_loss(
    model: LinearScorer,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(model(features), targets)


def pairwise_loss(model: LinearScorer, preferred: torch.Tensor, nonpreferred: torch.Tensor) -> torch.Tensor:
    return F.softplus(-(model(preferred) - model(nonpreferred))).mean()


def listnet_target_distribution(
    probabilities: torch.Tensor,
    temperature: float = 1.0,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("Listwise temperature must be positive")
    if not 0 < epsilon < 0.5:
        raise ValueError("Listwise logit epsilon must be in (0, 0.5)")
    clipped = probabilities.clamp(epsilon, 1.0 - epsilon)
    logits = torch.log(clipped) - torch.log1p(-clipped)
    return torch.softmax(logits / temperature, dim=0)


def listwise_loss(
    model: LinearScorer,
    frame: pd.DataFrame,
    temperature: float = 1.0,
    logit_epsilon: float = 1e-4,
) -> torch.Tensor:
    losses = []
    device = _model_device(model)
    features = _tensor_features(frame).to(device)
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    for _, group in frame.groupby("job_id", sort=True):
        index = torch.tensor(
            positions.loc[group.index].to_numpy(int),
            dtype=torch.long,
            device=device,
        )
        targets = torch.tensor(
            group["y_prob"].to_numpy(float),
            dtype=torch.float32,
            device=device,
        )
        target_distribution = listnet_target_distribution(
            targets, temperature, logit_epsilon
        )
        log_prediction = torch.log_softmax(model(features[index]), dim=0)
        losses.append(-(target_distribution * log_prediction).sum())
    if not losses:
        raise ValueError("Listwise loss requires at least one query")
    return torch.stack(losses).mean()


def _prepared_listwise_loss(
    model: LinearScorer,
    prepared: PreparedDataset,
    query_ids,
    temperature: float,
    logit_epsilon: float,
) -> torch.Tensor:
    losses = []
    for query_id in query_ids:
        index = prepared.query_indices[query_id]
        target_distribution = listnet_target_distribution(
            prepared.targets[index], temperature, logit_epsilon
        )
        log_prediction = torch.log_softmax(model(prepared.features[index]), dim=0)
        losses.append(-(target_distribution * log_prediction).sum())
    if not losses:
        raise ValueError("Listwise loss requires at least one query")
    return torch.stack(losses).mean()


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    model: LinearScorer,
    loss: torch.Tensor,
    gradient_clip_norm: float,
) -> None:
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()


def _listwise_query_batches(
    frame: pd.DataFrame,
    queries_per_batch: int,
    query_ids: np.ndarray | None = None,
):
    if queries_per_batch <= 0:
        raise ValueError("Listwise queries per batch must be positive")
    ordered = (
        np.asarray(query_ids)
        if query_ids is not None
        else np.asarray(sorted(frame["job_id"].unique()))
    )
    for start in range(0, len(ordered), queries_per_batch):
        query_batch = set(ordered[start:start + queries_per_batch])
        yield frame[frame["job_id"].isin(query_batch)]


def _query_batches(query_ids: np.ndarray, queries_per_batch: int):
    if queries_per_batch <= 0:
        raise ValueError("Listwise queries per batch must be positive")
    for start in range(0, len(query_ids), queries_per_batch):
        yield query_ids[start:start + queries_per_batch]


def _run_training_epoch(
    formulation: str,
    model: LinearScorer,
    optimizer: torch.optim.Optimizer,
    train: PreparedDataset,
    train_pair_tensors: tuple[torch.Tensor, torch.Tensor] | None,
    batch_size: int,
    listwise_temperature: float,
    listwise_logit_epsilon: float,
    gradient_clip_norm: float,
    rng: np.random.RandomState,
) -> None:
    model.train()
    if formulation == "pointwise":
        for indices in _batches(len(train.features), batch_size, rng):
            loss = _pointwise_tensor_loss(
                model, train.features[indices], train.targets[indices]
            )
            _optimizer_step(optimizer, model, loss, gradient_clip_norm)
    elif formulation == "pairwise":
        preferred, nonpreferred = train_pair_tensors
        for indices in _batches(len(preferred), batch_size, rng):
            loss = pairwise_loss(model, preferred[indices], nonpreferred[indices])
            _optimizer_step(optimizer, model, loss, gradient_clip_norm)
    elif formulation == "listwise":
        query_ids = np.asarray(train.query_order, dtype=object)
        rng.shuffle(query_ids)
        for batch in _query_batches(query_ids, batch_size):
            loss = _prepared_listwise_loss(
                model, train, batch, listwise_temperature,
                listwise_logit_epsilon,
            )
            _optimizer_step(optimizer, model, loss, gradient_clip_norm)
    else:
        raise ValueError(f"Unknown formulation: {formulation}")


def _objective_loss(
    formulation: str,
    model: LinearScorer,
    prepared: PreparedDataset,
    pair_tensors: tuple[torch.Tensor, torch.Tensor] | None,
    listwise_temperature: float,
    listwise_logit_epsilon: float,
) -> float:
    if formulation == "pointwise":
        loss = _pointwise_tensor_loss(model, prepared.features, prepared.targets)
    elif formulation == "pairwise":
        loss = pairwise_loss(model, *pair_tensors)
    else:
        loss = _prepared_listwise_loss(
            model, prepared, prepared.query_order,
            listwise_temperature, listwise_logit_epsilon,
        )
    return float(loss.item())


def _build_pairwise_state(
    formulation: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    pair_delta: float,
    max_pairs_per_job: int,
    seed: int,
    device: str | torch.device = "cpu",
) -> PairwiseState | None:
    if formulation != "pairwise":
        return None
    train_table = build_pair_table(train, pair_delta, max_pairs_per_job, seed)
    validation_table = build_pair_table(
        validation, pair_delta, max_pairs_per_job, seed + 1
    )
    return PairwiseState(
        train_table=train_table,
        validation_table=validation_table,
        train_tensors=_pair_tensors(train, train_table, device),
        validation_tensors=_pair_tensors(validation, validation_table, device),
        train_hash=pair_table_hash(train_table),
        validation_hash=pair_table_hash(validation_table),
    )


def _weak_validation_ndcg_at_5(
    model: LinearScorer,
    prepared: PreparedDataset,
) -> float:
    scores = predict_scores(model, prepared)
    relevance = prepared.targets.cpu().numpy()
    values = []
    for query_id in prepared.query_order:
        index = prepared.query_indices[query_id].cpu().numpy()
        query_relevance = relevance[index]
        query_scores = scores[index]
        order = np.argsort(-query_scores, kind="mergesort")
        ideal_order = np.argsort(-query_relevance, kind="mergesort")
        discounts = np.log2(np.arange(2, min(5, len(index)) + 2))
        actual = np.sum(
            (np.power(2.0, query_relevance[order][:5]) - 1.0) / discounts
        )
        ideal = np.sum(
            (np.power(2.0, query_relevance[ideal_order][:5]) - 1.0) / discounts
        )
        values.append(0.0 if ideal == 0.0 else float(actual / ideal))
    if not values:
        raise ValueError("Weak-validation nDCG@5 requires at least one query")
    return float(np.mean(values))


def _batch_unit(formulation: str) -> str:
    return {
        "pointwise": "cv_job_pairs",
        "pairwise": "preference_pairs",
        "listwise": "queries",
    }[formulation]


def _train_one(
    formulation: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    pair_delta: float,
    max_pairs_per_job: int,
    listwise_temperature: float,
    listwise_logit_epsilon: float,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 5.0,
    device: str = "cpu",
    prepared_train: PreparedDataset | None = None,
    prepared_validation: PreparedDataset | None = None,
    pairwise_state: PairwiseState | None = None,
) -> TrainingResult:
    resolved = _resolve_device(device)
    rng = set_seed(seed)
    model = LinearScorer().to(resolved)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_data = prepared_train or _materialize_dataset(train, resolved)
    validation_data = prepared_validation or _materialize_dataset(validation, resolved)
    if formulation == "pairwise" and pairwise_state is None:
        pairwise_state = _build_pairwise_state(
            formulation, train, validation, pair_delta,
            max_pairs_per_job, seed, resolved,
        )
    train_pair_tensors = pairwise_state.train_tensors if pairwise_state else None
    validation_pair_tensors = pairwise_state.validation_tensors if pairwise_state else None

    best_state = None
    best_ndcg = float("-inf")
    best_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    wait = 0
    final_epoch = 0
    history_rows = []
    for epoch in range(1, max_epochs + 1):
        final_epoch = epoch
        _run_training_epoch(
            formulation, model, optimizer, train_data, train_pair_tensors,
            batch_size, listwise_temperature, listwise_logit_epsilon,
            gradient_clip_norm, rng,
        )
        model.eval()
        with torch.no_grad():
            train_loss = _objective_loss(
                formulation, model, train_data, train_pair_tensors,
                listwise_temperature, listwise_logit_epsilon,
            )
            validation_loss = _objective_loss(
                formulation, model, validation_data, validation_pair_tensors,
                listwise_temperature, listwise_logit_epsilon,
            )
        validation_ndcg = _weak_validation_ndcg_at_5(model, validation_data)
        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "generalization_gap": validation_loss - train_loss,
            "validation_weak_ndcg_at_5": validation_ndcg,
        })
        improved = validation_ndcg > best_ndcg + 1e-12
        tied_better_loss = (
            abs(validation_ndcg - best_ndcg) <= 1e-12
            and validation_loss < best_loss - 1e-6
        )
        if improved or tied_better_loss:
            best_ndcg = validation_ndcg
            best_loss = validation_loss
            best_train_loss = train_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(
        formulation=formulation,
        model=model,
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        best_train_loss=best_train_loss,
        epochs_ran=final_epoch,
        stopped_early=final_epoch < max_epochs,
        weight_decay=float(weight_decay),
        gradient_clip_norm=float(gradient_clip_norm),
        selection_metric="validation_weak_ndcg_at_5",
        best_validation_ndcg_at_5=float(best_ndcg),
        batch_unit=_batch_unit(formulation),
        device=str(resolved),
        history=pd.DataFrame(history_rows),
        train_pair_hash=pairwise_state.train_hash if pairwise_state else None,
        validation_pair_hash=pairwise_state.validation_hash if pairwise_state else None,
    )


def select_hyperparameters(
    formulation: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
    learning_rates: list[float],
    batch_sizes: list[int],
    max_epochs: int,
    patience: int,
    pair_delta: float,
    max_pairs_per_job: int,
    listwise_temperature: float,
    listwise_logit_epsilon: float = 1e-4,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 5.0,
    device: str = "cpu",
) -> tuple[TrainingResult, pd.DataFrame]:
    resolved = _resolve_device(device)
    prepared_train = _materialize_dataset(train, resolved)
    prepared_validation = _materialize_dataset(validation, resolved)
    pairwise_state = _build_pairwise_state(
        formulation, train, validation, pair_delta,
        max_pairs_per_job, seed, resolved,
    )
    results = []
    best = None
    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            result = _train_one(
                formulation, train, validation, seed, float(learning_rate), int(batch_size),
                max_epochs, patience, pair_delta, max_pairs_per_job,
                listwise_temperature, listwise_logit_epsilon,
                weight_decay, gradient_clip_norm, str(resolved),
                prepared_train, prepared_validation, pairwise_state,
            )
            results.append(result.metadata())
            if (
                best is None
                or result.best_validation_ndcg_at_5
                > best.best_validation_ndcg_at_5 + 1e-12
                or (
                    abs(
                        result.best_validation_ndcg_at_5
                        - best.best_validation_ndcg_at_5
                    ) <= 1e-12
                    and result.best_validation_loss < best.best_validation_loss
                )
            ):
                best = result
    return best, pd.DataFrame(results)
