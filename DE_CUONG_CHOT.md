# Đề cương chốt

**Từ trọng số khớp thủ công đến Learning-to-Rank giám sát yếu cho CV–JD tiếng Việt**

> Thay thế `RESEARCH_DIRECTION_FINAL.md` và mọi bản trước.  
> Nguyên tắc: tư duy tuyến tính; core hẹp; khớp code hiện có; không phình phạm vi.

Chuỗi phương pháp (RankNet là bước cuối, không phải điểm xuất phát):

```
trọng số thủ công → học trọng số → LTR (RankNet) → ranking có tín hiệu skill-gap
```

---

## 0. Phát biểu nghiên cứu

Nghiên cứu kiểm tra liệu hệ khớp CV–JD tiếng Việt dựa trên trọng số đặc trưng thiết kế tay có thể được cải thiện tuần tự bằng (i) học tổ hợp đặc trưng, (ii) Learning-to-Rank giám sát yếu, và (iii) tín hiệu phụ skill-gap — trong điều kiện không có kết quả tuyển dụng thật.

Không khẳng định mô hình “học được bộ trọng số tối ưu” cho mọi đặc trưng nếu đầu ra là mạng phi tuyến.

---

## 1. Xuất phát: khớp theo trọng số thủ công

Baseline tham chiếu: Huynh et al. (2025, DEFI).

1. Vector đặc trưng bảng:  
   `x = [loc, skill, exp, role, desc]`
2. Điểm cố định:  
   `Score = 0.30·loc + 0.25·skill + 0.20·exp + 0.15·role + 0.10·desc`
3. Thường kèm ngưỡng nhị phân (ví dụ 0.45) → match / không match.

Dữ liệu làm việc cho paper run: subsample Kaggle 160 JD × 240 CV, tối đa 100 CV/JD (~16.000 cặp). Đây là protocol revision có kiểm soát từ mức ~80×120; vẫn không xử lý toàn bộ 14k×4k. Không có nhãn hired/rejected hay preference của nhà tuyển dụng.

**Hạn chế**

| # | Hạn chế | Hệ quả |
|---|---|---|
| 1 | Trọng số chủ quan, cố định cho mọi JD | Không phản ánh ưu tiên theo vị trí (junior/senior, skill-heavy/exp-heavy) |
| 2 | Phân loại ≠ xếp hạng | Tuyển dụng cần thứ tự trong pool, không chỉ nhãn nhị phân |
| 3 | Đánh giá phụ thuộc nguồn nhãn | Train/test cùng công thức → chỉ đo mức khớp công thức |

Định vị bài toán: **Learning-to-Rank giám sát yếu** — không xây hệ retrieve–rerank mới.

---

## 2. Bước 1 — Học tổ hợp đặc trưng (learned weighting)

Thay `w` thủ công bằng học tuyến tính trên cùng `x`:

```
H:  Score = w_tayᵀ x
B1: Score = w_họcᵀ x   (Logistic Regression, mục tiêu pointwise)
```

Câu hỏi trung gian: trên gold độc lập, B1 có hơn H không?

B1 chỉ là ablation. Không tính đóng góp.

---

## 3. Giới hạn của mục tiêu pointwise

Pointwise học mức phù hợp từng cặp `(CV, JD)`, không tối ưu trực tiếp quan hệ `CV_A ≻ CV_B | JD`.

Cần chuyển sang học xếp hạng theo cặp.

---

## 4. Bước 2 — Learning-to-Rank (RankNet)

RankNet (Burges et al., 2005): học hàm điểm `s = f_θ(x)` sao cho:

```
P(A ≻ B) = σ(s_A − s_B)
L_rank   = softplus(-(s_A − s_B))
```

- Đầu vào: đúng 5 đặc trưng bảng; `role`/`desc` dùng TF–IDF fit chỉ trên TRAIN rồi đóng băng khi transform validation/test/gold (không dùng BGE-M3 trong phạm vi này).
- Kiến trúc: MLP nhỏ. Linear head chỉ khi cần đọc hệ số.
- Không dùng margin loss cho mô hình chính.
- Thuật ngữ: *hàm xếp hạng / relevance function* — không gọi “học trọng số” nếu dùng MLP.

---

## 5. Thiếu nhãn tuyển dụng → giám sát yếu

Không có hiring outcome → không LTR có giám sát đầy đủ.

Pipeline (khớp repo):

```
đặc trưng → LF (+1 / 0 abstain / −1)
         → Dawid–Skene → y_prob
         → lấy mẫu cặp → RankNet
```

- LF: `lf_skill`, `lf_sem`, `lf_exp`, `lf_role`, `lf_loc`; ngưỡng chỉ fit trên train.
- Tổng hợp: Dawid–Skene; đối chứng majority/consensus.
- Phụ thuộc LF: tương quan / tỷ lệ đồng thuận (mô tả). Không dùng Fleiss’ κ như thước độc lập.
- `y_prob` là tín hiệu relevance yếu — không gọi nhãn đúng.
- B2: pointwise trên `y_prob` (ablation trước RankNet).
- Lấy mẫu cặp: trần `max_pairs_per_job`; hard-negative = lệch LF mạnh; cặp gần ngưỡng loại khỏi train.

---

## 6. Bước 3 — Ranking có tín hiệu skill-gap

Câu hỏi: mô hình phản ánh năng lực thiếu hụt hay chủ yếu độ tương đồng bề mặt?

```
L = L_rank + λ · L_gap
Gap = Skill_JD \ Skill_CV
```

Chỉ bắt kỹ năng thiếu hoàn toàn; không đo mức thành thạo; không gọi “xếp hạng giải thích được”.

Đối chứng: M1 (RankNet) so M2 (RankNet + skill-gap).

---

## 7. Câu hỏi nghiên cứu

**RQ1.** Trên gold graded, job-disjoint: chuỗi H → B1/B2 → M1 có cải thiện NDCG/MAP so với trọng số thủ công và baseline pointwise không?

**RQ2.** Thêm skill-gap có tăng tỷ lệ giữ thứ tự đúng dưới perturbation năng lực (so với M1) không?

Hai RQ độc lập; kết quả lệch nhau vẫn báo cáo đầy đủ.

---

## 8. Đóng góp

**C1.** Chứng minh chuỗi chuyển đổi có kiểm chứng khi thiếu hiring label:  
trọng số tay → học tổ hợp đặc trưng → LTR giám sát yếu (RankNet).  
Không đề xuất RankNet / weak supervision / đặc trưng mới.

**C2.** Kiểm tra auxiliary skill-gap (set-difference) có cải thiện hành vi xếp hạng theo năng lực không.

H, B1, B2 không phải đóng góp.

**Protocol đánh giá (không ngang C1/C2):** gold graded độc lập + perturbation tách yếu tố. Train dùng nhãn yếu; test không trùng nguồn sinh nhãn.

---

## 9. Phạm vi

| Trong phạm vi | Ngoài phạm vi |
|---|---|
| Subsample Kaggle 160 JD × 240 CV, 100 CV/JD | Toàn bộ 14k×4k; thu thập mới |
| 5 đặc trưng Huynh + TF–IDF | BGE-M3, BERT, BM25 bắt buộc |
| LF + Dawid–Skene | Gói Snorkel (trừ khi thật sự dùng) |
| RankNet ± skill-gap | LambdaMART bắt buộc; retrieve–rerank lớn |
| Gold 0–3, một annotator, job-disjoint | IAA đa annotator; GT chuyên gia tuyển dụng |
| NDCG@K, MAP@K, QualSens, bootstrap CI | Fairness; LLM làm nhãn train |

---

## 10. Thiết kế thực nghiệm

```
H  trọng số tay
      ↓
B1 học trọng số (pointwise BCE)
B2 pointwise trên y_prob
      ↓
M1 RankNet                 ← RQ1
      ↓
M2 RankNet + skill-gap     ← RQ2
```

| ID | Mục tiêu huấn luyện | Vai trò |
|---|---|---|
| H | Điểm heuristic cố định | Baseline |
| B1 | Pointwise BCE | Ablation — learned weighting |
| B2 | Pointwise soft (`y_prob`) | Ablation — nhãn yếu mềm |
| M1 | RankNet (cặp từ nhãn yếu) | Mô hình chính — RQ1 |
| M2 | RankNet + skill-gap | RQ2 |

Có thể thêm xếp hạng trực tiếp theo consensus/`y_prob` (không train) như kiểm tra phụ — không thay B2.

---

## 11. Đánh giá

### 11.1. Gold set

- 12–15 JD job-disjoint (không có trong train, không dùng fit LF/aggregator).
- Khoảng 100–150 cặp.
- Thang 0–3:

| Mức | Ý nghĩa |
|---|---|
| 3 | Rất phù hợp |
| 2 | Phù hợp, còn thiếu một phần |
| 1 | Liên quan yếu / một phần |
| 0 | Không phù hợp |

- Một annotator (tác giả), rubric cố định, rationale ngắn.  
  Gọi: *tập đánh giá graded do tác giả gán*. Không gọi ground truth chuyên gia.
- Metric: NDCG@5/10 và MAP@5/10; MAP nhị phân hóa `relevance >= 2`. CI chính là paired bootstrap trên `job_id` của graded set, không bootstrap từng cặp CV–JD.
- Gold Phase 3 gốc là 0–2 (`human_validated_benchmark.csv`). Bản đánh giá paper: `human_validated_benchmark_graded_0_3.csv` — map có kiểm soát từ aspect scores → `relevance` ∈ {0,1,2,3} (xem `data/gold/GRADE_0_3_MAPPING.md`). Không gọi ground truth.

### 11.2. Perturbation (RQ2)

Từ CV gốc V0 tạo riêng V_skill, V_exp, V_domain (không cộng dồn).  
`QualSens_x = P[score(V0) > score(V_x)]` — báo cáo tách theo nhóm.

### 11.3. Kiểm định

Paired bootstrap CI trên graded `job_id`; seed variability báo riêng. So sánh LLM pairwise: không bắt buộc; không dùng để train.

---

## 12. Thuật ngữ

| Dùng | Không dùng |
|---|---|
| Trọng số thủ công / heuristic cố định | “Học trọng số” cho H |
| Learned weighting | Chỉ B1 (hoặc linear head) |
| Hàm xếp hạng / relevance function | “Trọng số tối ưu” cho M1 MLP |
| Dawid–Skene; tín hiệu relevance yếu | Snorkel; “nhãn đúng” cho `y_prob` |
| Tương đồng ngữ nghĩa TF–IDF | BGE-M3 |
| RankNet + softplus / BCEWithLogits | Margin loss cho mô hình chính |
| Auxiliary skill-gap | Xếp hạng giải thích được |
| Tập đánh giá graded do tác giả gán | Expert / recruiter GT |
| Protocol perturbation chẩn đoán | Đóng góp / bộ dữ liệu mới |

---

## 13. Hạn chế

- Gold nhỏ, một annotator → thiên lệch chủ quan.
- Nhãn yếu thừa hưởng bias của LF; các LF có thể tương quan.
- Trích xuất skill và TF–IDF tiếng Việt giới hạn hiệu năng.
- Skill-gap không đo mức thành thạo trong kỹ năng đã có.
- Perturbation không thay dữ liệu hành vi tuyển dụng thật.
- Subsample không đại diện toàn bộ Kaggle.

---

## 14. Kế hoạch

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| 0–2 | Audit rò nhãn, LF, Dawid–Skene, tách job-disjoint | Đã làm |
| 3 | Khóa gold 0–3 | Cần chỉnh từ bản 0–2 |
| 4a | H / B1 / B2 / M1 trên gold → RQ1 | Tiếp theo |
| 4b | M2 + perturbation → RQ2 | Sau 4a |
| — | LLM pairwise / BGE / BERT | Ngoài đề cương |

---

## 15. Tài liệu neo

| Nhóm | Nguồn |
|---|---|
| Baseline | Huynh et al., 2025 (DEFI) |
| RankNet | Burges et al., 2005 |
| LTR tuyển dụng | Braun, 2017; Bukarina, 2019; Faliagka et al., 2014 |
| Ranking giám sát yếu | Dehghani et al., 2017; Lien et al., 2023 |
| Tổng hợp nhãn | Dawid–Skene |
| Perturbation | Wang et al., 2022; Piyavechvirat et al., 2026 |

Danh mục đầy đủ đưa vào phần Tài liệu tham khảo của báo cáo.
