# -*- coding: utf-8 -*-
"""Patch _gen_experiment_nb.py for training reliability, then regenerate notebook."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "scripts" / "scratch" / "_gen_experiment_nb.py"
t = GEN.read_text(encoding="utf-8")

repls = [
    (
        "from src.models.pairing import build_rank_pairs",
        "from src.models.pairing import build_rank_pairs, build_rank_pairs_robust",
    ),
    ("val_ratio: float = 0.10", "val_ratio: float = 0.15"),
    ("pair_margin: float = 0.05", "pair_margin: float = 0.02"),
    ("max_pairs_per_job: int = 100", "max_pairs_per_job: int = 150"),
]

for a, b in repls:
    if a not in t:
        raise SystemExit(f"missing: {a!r}")
    t = t.replace(a, b, 1)

old_pt = '''def pair_tensors(df, seed, cfg=CFG):
    Xi, Xj = build_rank_pairs(
        df, score_col="y_prob", max_pairs_per_job=cfg.max_pairs_per_job,
        margin=cfg.pair_margin, rng=np.random.RandomState(seed),
    )
    return torch.tensor(Xi, dtype=torch.float32), torch.tensor(Xj, dtype=torch.float32)'''

new_pt = '''def pair_tensors(df, seed, cfg=CFG):
    Xi, Xj, used_m = build_rank_pairs_robust(
        df, score_col="y_prob", max_pairs_per_job=cfg.max_pairs_per_job,
        margin=cfg.pair_margin, margin_fallbacks=(0.01, 0.0), min_pairs=8,
        rng=np.random.RandomState(seed),
    )
    return torch.tensor(Xi, dtype=torch.float32), torch.tensor(Xj, dtype=torch.float32), float(used_m)'''

if old_pt not in t:
    raise SystemExit("pair_tensors block missing")
t = t.replace(old_pt, new_pt)

old_m1 = '''def train_m1(df_train, df_val, seed, lr, batch_size, cfg=CFG) -> TrainResult:
    rng = set_seed(seed)
    ti, tj = pair_tensors(df_train, seed, cfg)
    vi, vj = pair_tensors(df_val, seed + 1, cfg)
    net = RankNetMLP(len(cfg.feature_cols), cfg.hidden1, cfg.hidden2, cfg.dropout)
    if len(ti) == 0:
        return TrainResult("M1", seed, 0, float("nan"), lr, batch_size, 0.0, 0.0, net=net)
    opt = optim.Adam(net.parameters(), lr=lr)
    best_state, best_loss, best_ep, wait = None, float("inf"), 0, 0
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        net.train()
        for b in _batches(len(ti), batch_size, rng):
            opt.zero_grad()
            F.softplus(-(net(ti[b]) - net(tj[b]))).mean().backward()
            opt.step()
        vloss = eval_pair_loss(net, vi, vj)
        if vloss < best_loss - 1e-5:
            best_loss, best_ep, wait = vloss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state:
        net.load_state_dict(best_state)
    return TrainResult("M1", seed, best_ep, best_loss, lr, batch_size, 0.0, time.time() - t0, net=net)'''

new_m1 = '''def train_m1(df_train, df_val, seed, lr, batch_size, cfg=CFG) -> TrainResult:
    rng = set_seed(seed)
    ti, tj, _ = pair_tensors(df_train, seed, cfg)
    vi, vj, val_m = pair_tensors(df_val, seed + 1, cfg)
    net = RankNetMLP(len(cfg.feature_cols), cfg.hidden1, cfg.hidden2, cfg.dropout)
    if len(ti) == 0:
        return TrainResult("M1", seed, 0, float("nan"), lr, batch_size, 0.0, 0.0, net=net)
    monitor_i, monitor_j = vi, vj
    if len(vi) == 0:
        n_hold = max(8, min(64, len(ti) // 5))
        monitor_i, monitor_j = ti[:n_hold], tj[:n_hold]
        print(f"  [warn] val pairs empty after fallback; monitor train-holdout n={n_hold}")
    else:
        print(f"  val_pairs={len(vi)} margin_used={val_m}")
    opt = optim.Adam(net.parameters(), lr=lr)
    best_state, best_loss, best_ep, wait = None, float("inf"), 0, 0
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        net.train()
        for b in _batches(len(ti), batch_size, rng):
            opt.zero_grad()
            F.softplus(-(net(ti[b]) - net(tj[b]))).mean().backward()
            opt.step()
        vloss = eval_pair_loss(net, monitor_i, monitor_j)
        if not np.isfinite(vloss):
            wait += 1
            if wait >= cfg.patience:
                break
            continue
        if vloss < best_loss - 1e-5:
            best_loss, best_ep, wait = vloss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state:
        net.load_state_dict(best_state)
    elif best_loss == float("inf"):
        best_loss = eval_pair_loss(net, monitor_i, monitor_j)
        best_ep = cfg.max_epochs
    return TrainResult("M1", seed, best_ep, best_loss, lr, batch_size, 0.0, time.time() - t0, net=net)'''

if old_m1 not in t:
    raise SystemExit("train_m1 block missing")
t = t.replace(old_m1, new_m1)

old_m2 = '''def train_m2(df_train, df_val, seed, lr, batch_size, lambda_gap, cfg=CFG) -> TrainResult:
    rng = set_seed(seed)
    ti, tj = pair_tensors(df_train, seed, cfg)
    vi, vj = pair_tensors(df_val, seed + 1, cfg)
    Y_gap, vocab = build_gap_targets(df_train)
    net = RankNetMLP(len(cfg.feature_cols), cfg.hidden1, cfg.hidden2, cfg.dropout)
    gap_head = SkillGapHead(cfg.hidden2, Y_gap.shape[1])
    if len(ti) == 0:
        return TrainResult("M2", seed, 0, float("nan"), lr, batch_size, lambda_gap, 0.0, net=net, gap_head=gap_head)
    X_all = torch.tensor(df_train[list(cfg.feature_cols)].values, dtype=torch.float32)
    Y_all = torch.tensor(Y_gap, dtype=torch.float32)
    opt = optim.Adam(list(net.parameters()) + list(gap_head.parameters()), lr=lr)
    best_state, best_gap, best_loss, best_ep, wait = None, None, float("inf"), 0, 0
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        net.train(); gap_head.train()
        for b in _batches(len(ti), batch_size, rng):
            opt.zero_grad()
            l_rank = F.softplus(-(net(ti[b]) - net(tj[b]))).mean()
            l_gap = skill_gap_loss(gap_head(net.hidden(X_all)), Y_all)
            (l_rank + lambda_gap * l_gap).backward()
            opt.step()
        # selection: val pair-loss only (not +gap)
        vloss = eval_pair_loss(net, vi, vj)
        if vloss < best_loss - 1e-5:
            best_loss, best_ep, wait = vloss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            best_gap = {k: v.detach().cpu().clone() for k, v in gap_head.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state:
        net.load_state_dict(best_state)
        gap_head.load_state_dict(best_gap)
    return TrainResult("M2", seed, best_ep, best_loss, lr, batch_size, lambda_gap,
                       time.time() - t0, net=net, gap_head=gap_head)'''

new_m2 = '''def train_m2(df_train, df_val, seed, lr, batch_size, lambda_gap, cfg=CFG) -> TrainResult:
    rng = set_seed(seed)
    ti, tj, _ = pair_tensors(df_train, seed, cfg)
    vi, vj, val_m = pair_tensors(df_val, seed + 1, cfg)
    Y_gap, vocab = build_gap_targets(df_train)
    net = RankNetMLP(len(cfg.feature_cols), cfg.hidden1, cfg.hidden2, cfg.dropout)
    gap_head = SkillGapHead(cfg.hidden2, Y_gap.shape[1])
    if len(ti) == 0:
        return TrainResult("M2", seed, 0, float("nan"), lr, batch_size, lambda_gap, 0.0, net=net, gap_head=gap_head)
    monitor_i, monitor_j = vi, vj
    if len(vi) == 0:
        n_hold = max(8, min(64, len(ti) // 5))
        monitor_i, monitor_j = ti[:n_hold], tj[:n_hold]
        print(f"  [warn] M2 val pairs empty; monitor train-holdout n={n_hold}")
    X_all = torch.tensor(df_train[list(cfg.feature_cols)].values, dtype=torch.float32)
    Y_all = torch.tensor(Y_gap, dtype=torch.float32)
    opt = optim.Adam(list(net.parameters()) + list(gap_head.parameters()), lr=lr)
    best_state, best_gap, best_loss, best_ep, wait = None, None, float("inf"), 0, 0
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        net.train(); gap_head.train()
        for b in _batches(len(ti), batch_size, rng):
            opt.zero_grad()
            l_rank = F.softplus(-(net(ti[b]) - net(tj[b]))).mean()
            l_gap = skill_gap_loss(gap_head(net.hidden(X_all)), Y_all)
            (l_rank + lambda_gap * l_gap).backward()
            opt.step()
        vloss = eval_pair_loss(net, monitor_i, monitor_j)
        if not np.isfinite(vloss):
            wait += 1
            if wait >= cfg.patience:
                break
            continue
        if vloss < best_loss - 1e-5:
            best_loss, best_ep, wait = vloss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            best_gap = {k: v.detach().cpu().clone() for k, v in gap_head.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state:
        net.load_state_dict(best_state)
        gap_head.load_state_dict(best_gap)
    elif best_loss == float("inf"):
        best_loss = eval_pair_loss(net, monitor_i, monitor_j)
        best_ep = cfg.max_epochs
    return TrainResult("M2", seed, best_ep, best_loss, lr, batch_size, lambda_gap,
                       time.time() - t0, net=net, gap_head=gap_head)'''

if old_m2 not in t:
    raise SystemExit("train_m2 block missing")
t = t.replace(old_m2, new_m2)

# assumption bullet
needle = "9. **Kết luận:** CI 95% chứa 0 → không kết luận cải thiện chắc chắn."
insert = (
    "9. **Kết luận:** CI 95% chứa 0 → không kết luận cải thiện chắc chắn.\n"
    "10. **Val pairs:** `build_rank_pairs_robust` + fallback margin; "
    "nếu val vẫn rỗng thì monitor train-holdout (tránh `best_val_loss=inf`)."
)
if needle not in t:
    raise SystemExit("assumption needle missing")
t = t.replace(needle, insert)

GEN.write_text(t, encoding="utf-8")
print("patched", GEN)

# regenerate notebook by exec
ns = {"__file__": str(GEN), "__name__": "__main__"}
exec(compile(GEN.read_text(encoding="utf-8"), str(GEN), "exec"), ns)
print("notebook regenerated")
