# Research Audit — Experimental Validity

Reviewer stance: scientific validity > điểm số cao.

Nghiên cứu kiểm tra chuỗi  
`H → B1 → B2 → M1 → M2`  
có cải thiện thứ tự CV–JD khi **không có hiring label** — không chứng minh “mô hình tốt nhất / production / dự đoán tuyển dụng”.

---

## Verdict tổng

| Hạng mục | Đạt? | Ghi chú |
|---|---|---|
| RQ rõ, trong phạm vi | Có | RQ1 ranking trên gold; RQ2 QualSens |
| Pipeline khớp RQ | Có | Đúng chuỗi; không BERT/BGE/LambdaMART |
| Job-disjoint + gold hold-out | Có | Bắt buộc giữ |
| Metric ranking | Có | NDCG/MAP; không dùng Acc/F1 làm chính |
| Multi-seed + bootstrap | Có | 5 seeds; CI trên job |
| Label independence | **Yếu** | LF = ngưỡng trên cùng 5 feature đầu vào |
| B1 vs H công bằng | **Yếu** | B1 học từ `heuristic_label` cắt từ công thức H |
| Model selection | **Yếu** | Early-stop/HP từng dùng NDCG(`y_prob`) |
| Gold 0–3 | **Chấp nhận có điều kiện** | Map từ 0–2; grade 3 chỉ 6/100 |
| Kết luận | Phải thận trọng | CI chứa 0 → không claim chắc |

---

## Critical / Major

### C1 — Feature–label dependence (weak supervision)
LF (`lf_skill`, `lf_sem`, …) là ngưỡng của chính `skill_iou`, `desc_sem_sim`, … rồi model lại nhận cùng vector `x`.  
`y_prob` **không** độc lập với đặc trưng.

**Hệ quả:** RankNet có thể học lại cấu trúc LF, không phải “relevance ngoài feature”.  
**Xử lý bắt buộc:** báo cáo coverage / conflict / correlation LF; nhấn `y_prob ≠ ground truth`; đánh giá **chỉ** trên gold độc lập; không claim LF là nguồn tri thức ngoài feature.

### C2 — B1 target trùng nguồn với H
`heuristic_label = 1[H_score ≥ 0.45]`. B1 (logistic trên `x`) học lại bài cắt nhị phân của H.

**Hệ quả:** B1 > H trên gold vẫn có ý (khả năng generalize của trọng số học), nhưng **không** chứng minh “nhãn độc lập”.  
**Xử lý:** giữ B1 đúng đề cương (ablation learned weighting); trong interpretation ghi rõ target phụ thuộc H; không gọi B1 là contribution.

### C3 — Model selection bằng NDCG(`y_prob`)
Chọn epoch/lr/batch theo NDCG trên val với target = `y_prob` → tối ưu khớp nhãn yếu.

**Xử lý:** early-stopping / HP theo **val RankNet pair-loss** (cùng objective train). Gold **không** dùng chọn model.

### C4 — B2 loss lệch spec
Spec: soft target + `BCEWithLogitsLoss`. Code cũ: Ridge MSE.

**Xử lý:** B2 = linear score + `BCEWithLogits(s, y_prob)` (soft BCE).

### M1 — Gold 0–3 map, grade 3 hiếm (6/100)
Map có kiểm soát từ aspect; một annotator; không IAA. Bootstrap ~12 JD → CI rộng.

**Xử lý:** dùng `relevance` 0–3; báo phân phối; không gọi GT; nếu CI chứa 0 → không kết luận mạnh.

### M2 — QualSens ≠ “hiểu năng lực” đầy đủ
QualSens = P[score(V0)>score(V_x)] sau perturbation feature.  
Nếu M2 ↑ NDCG nhưng QualSens không ↑: kết luận đúng là *cải thiện ranking metric, chưa chứng minh robustness năng lực*.

### M3 — MAP@K
Protocol paper định nghĩa relevant khi `relevance ≥ 2`; `map_at_k` và artifact authoritative đã khóa đúng ngưỡng này. NDCG tiếp tục dùng đầy đủ graded 0–3.

---

## Minor (giữ / nêu limitation)

- Paper protocol revision: subsample 160 JD × 240 CV, tối đa 100 CV/JD (~16.000 cặp) — vẫn không phải toàn corpus Kaggle.
- Skill-gap fallback `1 - skill_iou` nếu thiếu skill thô.
- Không kiểm tra near-duplicate CV giữa split (job-disjoint đã chặn leakage JD).
- Không Fleiss’ κ như test độc lập LF.

---

## Quyết định triển khai notebook (sau audit)

1. Giữ chuỗi H/B1/B2/M1/M2 và phạm vi đề cương.  
2. Đổi B2 → soft BCEWithLogits.  
3. Early-stop / HP → val `L_rank`.  
4. Section bắt buộc: hypothesis, assumptions, dataset audit, leakage check, LF audit, multi-seed, bootstrap, interpretation (không overclaim).  
5. Scientific validity > chasing NDCG.

---

## Follow-up audit (external review) — đã xử lý 2026-08-23

| Issue | Xử lý |
|---|---|
| **C-1** CI neo seed cuối | Đã supersede: CI **chính** = paired bootstrap trên graded `job_id`, sau khi average per-job metric qua seeds. Seed variability báo riêng. |
| **C-2** TF–IDF fit trước split | Đã supersede: authoritative `TrainOnlyFeaturePipeline` fit vocabulary/IDF chỉ từ unique TRAIN JD/CV; validation/test/gold chỉ transform bằng state đóng băng. |
| **M-2** markdown λ | Sửa: λ chọn theo val pair-loss **thuần** (không +gap). |
| **M-4** thiếu B1↔B2 | Thêm `B2_vs_B1` / `B1_vs_B2` vào bảng so sánh. |
| **M-5** seed kép | Ghi rõ trong §02 / §11: seed vừa split vừa init; within-seed HP fair. |
| **M-1** gold nhỏ | Giữ disclosure; nhấn trong §15/§17 (power thấp). |

### Reliability patch (2026-08-23, sau paper run)

- `pair_margin` 0.05 → **0.02**; `val_ratio` 0.10 → **0.15**; `max_pairs_per_job` 150.
- `build_rank_pairs_robust`: fallback margin 0.01 → 0.0; nếu val vẫn rỗng thì fail-fast, không tune/early-stop trên train. Artifact lưu `monitor_source`, `val_margin_used`, `n_monitor_pairs`.
- `parse_skill_set`: lọc token rác (`..`, `...`, punctuation-only).
- **Không** mở rộng gold (cần annotate tay); không claim kết quả mới cho đến khi re-run notebook.

### Correctness repair (2026-08-23, phải re-run trước khi khóa paper numbers)

1. **Label model train-only.** Pipeline cũ gọi `fit_predict()` trong mỗi lần aggregate, nên validation và graded evaluation set đã tự ước lượng lại prior/sensitivity/specificity. API mới fit Dawid–Skene đúng một lần trên train; mọi split còn lại chỉ dùng tham số đóng băng. Artifact `label_model_parameters_by_seed.json` lưu tham số theo seed, và notebook assert tham số không đổi sau validation/gold inference.
2. **Abstain-aware M-step.** Mẫu số sensitivity/specificity cũ gồm cả các dòng LF abstain. M-step mới dùng active mask riêng cho từng LF (`LF != 0`); abstain tiếp tục không đóng góp E-step likelihood. Regression tests kiểm tra mẫu số và posterior all-abstain.
3. **Khóa feature contract.** `FEATURE_COLS` từng trôi thành 7 cột do đưa hai tỷ lệ skill diagnostic vào model. Contract đã khôi phục đúng 5 cột của đề cương: `loc_match`, `skill_iou`, `exp_score`, `role_match`, `desc_sem_sim`. Các tỷ lệ skill và experience diagnostics vẫn tồn tại để phân tích lỗi nhưng không đi vào H/B1/B2/M1/M2 tensors.
4. **Runtime source traceability.** Notebook ghi path và SHA-256 rút gọn của `src.data.loader`, `src.weak.aggregator`, `src.models.skill_gap`; preflight fail nếu feature contract, train-only API hoặc parser sanitation không đúng. Artifact `runtime_preflight.json` ngăn việc upload notebook nhưng chạy nhầm `src/` cũ trên Jupyter server.
5. **Pair/gap audit.** Artifact `weak_score_pair_audit_by_seed.csv` lưu unique-score/tie diagnostics theo split. `results_all_runs.csv` lưu train/validation pair count, margin thực dùng, vocabulary size, target prevalence và M2 gap loss tại best/final epoch.

Mọi artifact và metric sinh trước correction này chỉ là lịch sử audit, không dùng làm kết luận cuối cho RQ1/RQ2. Correctness repair không được diễn giải là đảm bảo nDCG tăng.

### Dataset-scale revision (2026-08-23)

- Paper run tăng từ 80 JD × 120 CV, 70 CV/JD lên **160 JD × 240 CV, tối đa 100 CV/JD** (~16.000 cặp).
- Chỉ thay đổi quy mô đầu vào; giữ nguyên năm feature, H/B1/B2/M1/M2, `max_pairs_per_job=150`, split, seeds, objective, HP/λ grids và graded set.
- Graded set vẫn chỉ có 100 cặp trên khoảng 12 JD với một annotator; tăng weak-label data không làm tăng trực tiếp statistical power của evaluation.
- Kết quả ở quy mô 80×120 chỉ là lịch sử audit và không được trộn với paper numbers sau revision.

### Graded-ID namespace correction (2026-08-23)

- Phase 3 gán `JOB_xx`/`CV_xxx` theo **raw row index** trong hai CSV Kaggle. Loader trước đây tái sử dụng cùng format ID cho vị trí sau sampling, nên graded inference và gold-JD exclusion có thể trỏ nhầm thực thể.
- Loader mới giữ `_source_index` khi sample và xuất ID theo raw row index. Authoritative feature pipeline truy xuất graded rows theo raw ID nhưng chỉ dùng TF–IDF đã fit trên TRAIN.
- CLI fail-fast nếu raw ID không còn khớp `job_title`/`desired_job`, yêu cầu graded coverage 100/100, đồng thời loại toàn bộ graded JD **và graded CV** khỏi weak-label pool và preprocessing fit scope.
- Mọi metric graded sinh trước namespace correction này không dùng cho kết luận cuối.

### Authoritative paper protocol correction (2026-08-23)

- Lệnh duy nhất: `python scripts/run_paper_experiment.py --config configs/experiment.yaml`; `--smoke` giữ nguyên 160×240×100 nhưng giảm seed/grid/epoch.
- `global_skill_df.json`, phase scripts cũ, notebook implementation và các report cũ không được dùng cho paper numbers.
- Pair table enumerate unordered pair trong cùng JD, giữ `abs(y_prob_i-y_prob_j) >= 0.02`, cap 150/JD và ưu tiên deterministic LF-conflict hard negatives. M1/M2 dùng cùng pair-table hash.
- Gold chỉ được transform/chấm điểm sau model selection; MAP dùng `relevance >= 2`; CI chính paired-bootstrap theo graded JD.
- Smoke authoritative đã tạo đủ output tree dưới `results/`; đây là kiểm tra tích hợp, không phải paper numbers. Full run mới được dùng để kết luận RQ1/RQ2.
