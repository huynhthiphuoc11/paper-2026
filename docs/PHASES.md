# Checklist giai đoạn (theo DE_CUONG_CHOT)

| Phase | Mục tiêu | Artifact chính | Entry |
|---|---|---|---|
| 0–2 | Audit leakage, LF, Dawid–Skene, split job-disjoint | `data/splits/`, `data/weak_labels/`, `reports/phase*` | `scripts/audit/` |
| 3 | Khóa gold graded **0–3**, job-disjoint | `data/gold/` | `scripts/gold/` |
| 4a | H / B1 / B2 / M1 trên graded set → RQ1 + paired-JD CI | `results/predictions`, `results/metrics`, `results/tables` | `python scripts/run_paper_experiment.py --config configs/experiment.yaml` |
| 4b | M2 + isolated perturbation → RQ2 | `results/metrics/perturbation_metrics.csv`, `results/tables/perturbation_results.csv` | cùng authoritative command; notebook chỉ trình bày artifacts |

## Chuỗi mô hình

```
H  trọng số tay
B1 learned weighting (pointwise BCE)
B2 pointwise soft y_prob
M1 RankNet                         ← RQ1
M2 RankNet + skill-gap             ← RQ2
```

## Quy tắc

- Fit LF / aggregator **chỉ trên train**.
- Gold job-disjoint: không dùng để train hay tune.
- Không gọi RankNet là “học trọng số” (trừ linear head).
- Không thêm BGE/BERT/BM25 trong phạm vi đề cương chốt.
