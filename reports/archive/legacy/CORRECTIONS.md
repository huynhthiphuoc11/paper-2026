# Chấn Chỉnh Chi Tiết: 5 Điểm Ambiguity Cần Fix

---

## 1. ❌ Vấn Đề: Fleiss' kappa vs. Kendall's tau_b (Mục 3 + Mục 6)

### Tình Trạng Hiện Tại (Sai)
- Mục 3: "Đo mức đồng thuận giữa 3 nguồn bằng **Fleiss' kappa**"
- Mục 6: "Đo đồng thuận giữa người chú thích bằng **Kendall's tau_b**"

### Sai Lầm
**Fleiss' kappa** dùng khi: nhiều rater, cùng một tập items, rater chọn từ các nhãn rời rạc (categorical).
- Ở đây: 3 weak signals (S1, S2, S3) **không phải 3 rater** mà là **3 loại thông tin khác nhau**.
- Fleiss' kappa sẽ đo "3 nguồn bao nhiêu % cho cùng một câu trả lời" — đó là *độ đồng thuận / agreement*, không phải *tính độc lập / independence*.

**Kendall's tau_b** dùng khi: **ranking/ordinal data** — có thứ tự. Nhưng trong Gold set annotation (mục 6), mỗi (CV, Job) được chú thích `relevant / irrelevant` = **binary categorical**, không phải ranking.

### ✓ Sửa Thành

#### Mục 3 — Đo Độ "Đúng" Của 3 Weak Signals (không dùng Fleiss' kappa)

Không đo "mức đồng thuận" (agreement) mà đo **độ chính xác từng signal** so với dev-gold (nhãn người chú thích 500–1000 cặp):

```
Cho mỗi signal S_i (i ∈ {1,2,3}):
  Tính Precision_i = TP_i / (TP_i + FP_i)
                   = (số cặp S_i nói "match" mà dev-gold cũng nói "match") 
                     / (tổng số cặp S_i nói "match")
  Tính Recall_i    = TP_i / (TP_i + FN_i)
                   = (số cặp S_i nói "match" mà dev-gold cũng nói "match")
                     / (tổng số cặp dev-gold nói "match")
  Tính F1_i        = 2 * Precision_i * Recall_i / (Precision_i + Recall_i)
```

**Mục tiêu**: Precision_i > 85% cho mỗi i. Điều này chứng minh: "signal này nếu nói match thì thường đúng", không chứng minh "3 signals luôn đồng ý" — đó là hai câu hỏi khác nhau.

**Nếu muốn đo độ đồng thuận giữa 3 signals** (riêng, không phải mục tiêu chính):
- Dùng **Cramér's V** (symmetric measure of association, 0–1) hoặc **Cohen's kappa** cộng dồn trên từng cặp signal.
- Cách dễ hơn: tính % cặp mà 3 signals cho cùng kết quả (majority vote) → con số % này, không phải thống kê đặc biệt gì.
- **Nhưng không gọi đây là "Fleiss' kappa"** vì Fleiss' kappa là một công thức cụ thể, không phải "% đồng ý".

#### Mục 6 — Đo Độ Đồng Thuận Giữa Người Chú Thích

Gold set có **2–3 người chú thích độc lập** (tùy budget, tối thiểu 2):

- Nếu **binary labels** (`relevant / irrelevant`): dùng **Cohen's kappa** (2 rater) hoặc **Fleiss' kappa** (≥3 raters).
  - Giải thích: Cohen's kappa = (observed agreement - expected agreement by chance) / (1 - expected agreement by chance)
  - Mục tiêu: kappa ≥ 0.7 (moderate-to-substantial agreement).

- Nếu **ranking** (CV1 > CV2 > CV3 > …): dùng **Kendall's tau_b** hoặc **Spearman's rho**.
  - Nhưng annotation kiểu ranking sẽ **rất khó** (người chú thích phải sort 10–20 CV per job).
  - **Khuyến cáo**: giữ binary labels, dùng Cohen's kappa thôi.

### 📝 Kết Luận Sửa

- **Xóa Fleiss' kappa khỏi Mục 3** → thay bằng tính Precision/Recall/F1 từng S_i so với dev-gold.
- **Giữ Cohen's kappa cho Mục 6** (hoặc Fleiss' nếu ≥3 annotators).
- **Xóa Kendall's tau_b khỏi Mục 6** (nó dành cho ranking, không dành cho binary labels).

---

## 2. ❌ Vấn đề: "Hard Negative" Định Nghĩa Mơ Hồ (Mục 4)

### Tình Trạng Hiện Tại (Sai)

```
| 0 | **Hard negative** | Cả 3 nguồn đồng thuận non-match, **hoặc** có 
                          mismatch mạnh một-chiều (một nguồn rất cao, 
                          nguồn khác rất thấp — tín hiệu mâu thuẫn rõ ràng) 
```

### Sai Lầm

- "một nguồn **rất cao**" = bao nhiêu? 0.8? 0.9?
- "nguồn khác **rất thấp**" = bao nhiêu? 0.2? 0.3?
- "mâu thuẫn **rõ ràng**" = gap >= bao nhiêu? 0.5? 0.6?

→ Không rõ → khi implement sẽ thường chọn tùy tiện → kết quả không reproducible.

### ✓ Sửa Thành

```
| Vote | Nhóm | Định Nghĩa | Công Thức Cụ Thể |
|------|------|-----------|------------------|
| 0 | **Clear Non-match** | Cả 3 nguồn nói non-match | S1 ≤ θ_s AND S2 ≤ θ_sk AND S3 ≤ θ_e |
| 3 | **Clear Relevant** | Cả 3 nguồn nói match | S1 > θ_s AND S2 > θ_sk AND S3 > θ_e |
| 1–2 ambiguous, nhưng thêm điều kiện: | **Strong Contradiction** | Một nguồn rất cao (>= θ_upper), ít nhất một nguồn rất thấp (< θ_lower), và chênh lệch >= Δ | Ví dụ: S1 ≥ 0.85 AND (S2 < 0.4 OR S3 < 0.4) AND (S1 - min(S2,S3)) ≥ 0.5 → **Giữ làm hard negative** |
| 1–2, gần ngưỡng, KHÔNG mismatch mạnh | **Uncertain** | Các nguồn gần ngưỡng θ, chênh lệch nhỏ | |Tất cả θ_lower ≤ S_i ≤ θ_upper → **Loại khỏi train** |

Cụ thể hơn — ví dụ bằng số:

```python
# Giả sử θ_s = 0.5, θ_sk = 0.6, θ_e = 0.7
# θ_upper = 0.85, θ_lower = 0.4, Δ = 0.5

def get_label(S1, S2, S3):
    votes = sum([S1 > θ_s, S2 > θ_sk, S3 > θ_e])  # 0, 1, 2, 3
    
    if votes == 3:
        return 2, "relevant"
    elif votes == 0:
        return 0, "clear_non_match"
    elif votes == 1 or votes == 2:
        # Check strong contradiction
        max_s = max(S1, S2, S3)
        min_s = min(S1, S2, S3)
        if max_s >= 0.85 and min_s < 0.4 and (max_s - min_s) >= 0.5:
            return 0, "hard_negative"  # mâu thuẫn mạnh → dùng làm negative
        # Check uncertain
        elif all(0.4 <= si <= 0.85 for si in [S1, S2, S3]):
            return None, "uncertain"  # loại khỏi
        else:
            return 1, "soft_ambiguous"  # edges → giữ, dùng làm positive với thấp confidence

    return None, "undefined"
```

### 📝 Kết Luận Sửa

- Thêm bảng 2D cụ thể với **threshold số** (0.85, 0.4, 0.5) trong Mục 4.
- **Code example** như trên, không chỉ mô tả chữ.
- Ghi rõ: "Các threshold này **phải được xác định trên dev-gold**" — không lấy tùy tiện. Sau khi tạo train labels, chạy lại trên dev-gold xem Precision của "hard_negative" có >= 85% không.

---

## 3. ❌ Vấn Đề: Pair Sampling "Có Kiểm Soát" Không Rõ (Mục 5)

### Tình Trạng Hiện Tại (Sai)

> "không lấy tích Cartesian đầy đủ $P^+ \times P^-$ (sẽ nổ số lượng và lệch phân bố theo job có nhiều CV). Thay vào đó: lấy mẫu có kiểm soát — số cặp mỗi job giới hạn theo $\min(|P^+|, |P^-|, k_{max})$, đảm bảo mỗi job đóng góp lượng cặp tương đương nhau vào tập train."

### Sai Lầm

- "$\min(|P^+|, |P^-|, k_{max})$" là cái gì? Số job? Số cặp?
- "giới hạn" = cứng hay mềm (probabilistic)?
- Nếu $|P^+| = 3, |P^-| = 50, k_{max} = 10$ → tạo $3 \times 10 = 30$ cặp (lấy all 3 positives, sample 10 negatives) hay sample cả positives + negatives?
- Xử lý imbalance thế nào (khi $|P^+| >> |P^-|$ hoặc ngược lại)?

### ✓ Sửa Thành

```python
# Pair sampling strategy cho Pairwise LTR

k_max = 100  # max cặp per job (tunable parameter)

for job in jobs:
    P_plus = [cv for cv in CVs_of_job if label[cv, job] == 2]  # relevant
    P_minus = [cv for cv in CVs_of_job if label[cv, job] == 0]  # hard-negative
    
    # Nếu quá ít dữ liệu, bỏ job này
    if len(P_plus) < 1 or len(P_minus) < 1:
        continue
    
    # Strategy 1: Hard limit (RECOMMENDED)
    # —————————————————————————————————
    # Tạo mọi cặp từ P_plus × P_minus, sau đó lấy mẫu K cặp ngẫu nhiên
    all_pairs = [(cv_p, cv_n) for cv_p in P_plus for cv_n in P_minus]
    
    K = min(len(all_pairs), k_max)  # limit tối đa k_max cặp/job
    sampled_pairs = random.sample(all_pairs, K)
    
    # Thêm vào training set
    for (cv_p, cv_n) in sampled_pairs:
        train_pairs.append({
            'query': job,
            'doc_1': cv_p,
            'doc_2': cv_n,
            'label': 1  # doc_1 (cv_p) should rank higher than doc_2 (cv_n)
        })

# Kiểm soát
print(f"Tổng training pairs: {len(train_pairs)}")
print(f"Trung bình cặp/job: {len(train_pairs) / len(jobs):.1f}")
print(f"Min, Max cặp/job: {min(...), max(...)}")
# → Mục tiêu: phân bố không quá lệch (ratio max/min < 3 là được)
```

**Lựa chọn thứ 2** (nếu muốn hard negative mining — bỏ qua với bản nháp này):
```python
# Strategy 2: Weighted sampling (hard negative mining)
# ————————————————————————————————
# Score của từng cặp dùng aggregation của 3 weak signals
for (cv_p, cv_n) in all_pairs:
    s_p = (S1[cv_p, job] + S2[cv_p, job] + S3[cv_p, job]) / 3  # simple average
    s_n = (S1[cv_n, job] + S2[cv_n, job] + S3[cv_n, job]) / 3  # NOT dùng công thức cũ!
    
    margin[cv_p, cv_n] = s_p - s_n
    # Ưu tiên cặp có margin nhỏ (cv_p chỉ hơn cv_n chút chút) → khó học
    # Weight = 1 / (1 + margin)

# Lấy mẫu có trọng số
sampled_pairs = np.random.choice(all_pairs, size=K, 
                                 p=weights/weights.sum(), 
                                 replace=False)
```

### 📝 Kết Luận Sửa

- **Xóa mô tả mơ hồ** ở Mục 5.
- **Thêm pseudocode cụ thể** như trên (Strategy 1 là chủ yếu, Strategy 2 là optional ghi chú).
- Ghi rõ: "Aggregation function dùng **simple average** của 3 weak signals, không dùng công thức 5-trọng-số cũ".

---

## 4. ❌ Vấn Đề: Loss Function & Training Details (Mục 7.1)

### Tình Trạng Hiện Tại (Sai)

> "đơn giản nhất là **logistic regression / linear layer có trọng số học được** $g(x) = \sigma(w^T x + b)$ với $w$ được cập nhật qua gradient descent; phức tạp hơn (nếu muốn bắt tương tác phi tuyến giữa các feature) là **MLP nhỏ 1-2 lớp ẩn**."

### Sai Lầm

- MLP "1-2 lớp ẩn" nhưng:
  - Bao nhiêu neurons? 32? 64? 128?
  - Activation function gì? ReLU? Tanh?
  - Bao nhiêu dropout? Có L2 regularization không?
  - Optimizer gì? Adam? SGD?
  - Learning rate bao nhiêu?
  - Bao nhiêu epoch? Khi nào stop?
  - Validation set từ đâu? (không được dùng Gold set, không được dùng test set từ công thức cũ)

→ Thiếu chi tiết → không reproducible → người đọc sẽ thắc mắc hoặc implement sai.

### ✓ Sửa Thành

**7.1 Kiến Trúc Mô Hình**

```
Condition D (proposed) dùng:

Input:  x = [S1, S2, S3]  (3 weak signals)
        hoặc x = [S1, S2, S3, embedding_CV, embedding_Job]  
               (nếu variant D-embed)

Output: score ∈ ℝ (để rank CVs)

Option 1 — Linear Ranker (simplest):
  score = w_1 * S1 + w_2 * S2 + w_3 * S3 + b
  w_1, w_2, w_3, b: learned from training data

Option 2 — Small MLP (recommended):
  h = ReLU(W_1 @ x + b_1)           # shape: (64,)
  logits = W_2 @ h + b_2            # shape: (1,)
  score = logits.squeeze()           # ℝ
  
  Hyperparameter:
    - Hidden size = 64 (or 32/128 if tuning)
    - No dropout (train set nhỏ, dropout không cần)
    - L2 regularization λ = 0.001 (optional, để tránh overfit)
    - Activation: ReLU
```

**7.2 Training Setup**

```
Data split:
  - Train set: 80% weak-labeled data (generated from 3 signals)
  - Validation set: 20% weak-labeled data (same source)
    ⚠️  Validation set dùng để tuning hyperparameter (learning rate, λ)
        và early stopping, NOT dùng Gold set.

Optimizer: Adam
  - Learning rate: 0.001 (hoặc search [0.0001, 0.01])
  - Batch size: 32 (hoặc 16/64)
  - Weight decay (L2): 0.001

Loss function (for condition D - Pairwise):
  L_RankNet = log(1 + exp(-(s_i - s_j)))
    where (i, j) là cặp training từ mục 5
          s_i = f(cv_i, job), s_j = f(cv_j, job)
          và cv_i > cv_j (i.e., cv_i ∈ P+, cv_j ∈ P-)

Training loop:
  for epoch in 1:100:
    train_loss = 0
    for batch in train_loader:
      for (cv_i, cv_j, job) in batch:
        s_i = model(x_i)
        s_j = model(x_j)
        loss = log(1 + exp(-(s_i - s_j)))
        train_loss += loss
      
      # Backward + optimizer step
      backward(mean(batch_loss))
      optimizer.step()
      optimizer.zero_grad()
    
    # Evaluate on validation set
    val_loss = evaluate(val_loader)
    
    # Early stopping
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      patience_counter = 0
      save_model_checkpoint()
    else:
      patience_counter += 1
      if patience_counter > 10:
        break  # Stop training
  
  # Load best checkpoint từ validation
  model = load_model_checkpoint()
```

### 📝 Kết Luận Sửa

- **Thêm bảng Hyperparameter** cụ thể (learning rate, batch size, λ, hidden size).
- **Thêm training loop pseudocode** với early stopping rõ ràng.
- Ghi rõ: "Validation set từ weak-labeled data, **không dùng Gold set**".
- Nếu có ablation C (Condition C dùng BCE), viết riêng loss function cho nó:
  ```
  L_BCE = -[y * log(sigmoid(s)) + (1-y) * log(1 - sigmoid(s))]
    where y ∈ {0, 1}, s = f(cv, job)
  ```

---

## 5. ❌ Vấn Đề: Statistical Significance Test (Mục 9)

### Tình Trạng Hiện Tại (Sai)

> "vì số job trong Gold set nhỏ (10–15), **không dùng t-test tham số thông thường** — dùng **bootstrap resampling** hoặc **permutation test** trên chênh lệch NDCG giữa hai điều kiện để có khoảng tin cậy đáng tin cậy hơn với mẫu nhỏ."

### Sai Lầm

- Bootstrap resampling over what unit?
  - Resample jobs? Resample CV-job pairs? Resample NDCG scores?
- Bao nhiêu iterations?
- Confidence interval 95% = z-score hay percentile method?
- Có dùng **paired** test (cùng Gold set) hay **unpaired** test không?
- P-value tính thế nào từ bootstrap?

### ✓ Sửa Thành

**Vì Gold set có ~10–15 jobs**, mỗi job có ~10–20 CV pairs, **tổng 100–300 CV-job pairs** trong Gold set:

```python
# Bootstrap Paired Test cho so sánh hai phương pháp
# (Condition A vs. Condition D, hoặc bất kỳ cặp nào)

import numpy as np
from sklearn.metrics import ndcg_score

# Input:
# - gold_queries: list of 10-15 job IDs
# - gold_y_true: dict[job_id] = relevance labels (1 or 0) cho mỗi CV trong job
# - y_pred_A: dict[job_id] = scores từ method A (Condition A)
# - y_pred_D: dict[job_id] = scores từ method D (Condition D)

def compute_ndcg_at_k(y_true, y_pred, k=5):
    """Tính NDCG@k cho một query (job)"""
    return ndcg_score([y_true], [y_pred], k=k)

# Bước 1: Tính NDCG@5 cho từng method trên từng job
ndcg_A_per_job = {}
ndcg_D_per_job = {}

for job_id in gold_queries:
    y_true = gold_y_true[job_id]
    y_pred_a = y_pred_A[job_id]
    y_pred_d = y_pred_D[job_id]
    
    ndcg_A_per_job[job_id] = compute_ndcg_at_k(y_true, y_pred_a, k=5)
    ndcg_D_per_job[job_id] = compute_ndcg_at_k(y_true, y_pred_d, k=5)

# Bước 2: Tính chênh lệch trên từng job
differences = np.array([
    ndcg_D_per_job[job] - ndcg_A_per_job[job] 
    for job in gold_queries
])  # shape: (n_jobs,), e.g., (12,)

observed_diff = np.mean(differences)

print(f"Observed mean difference (D - A): {observed_diff:.4f}")
print(f"  Method A: {np.mean(list(ndcg_A_per_job.values())):.4f} ± {np.std(list(ndcg_A_per_job.values())):.4f}")
print(f"  Method D: {np.mean(list(ndcg_D_per_job.values())):.4f} ± {np.std(list(ndcg_D_per_job.values())):.4f}")

# Bước 3: Bootstrap paired test
n_bootstrap = 10000
bootstrap_diffs = []

for b in range(n_bootstrap):
    # Resample jobs WITH replacement
    bootstrap_jobs = np.random.choice(gold_queries, size=len(gold_queries), replace=True)
    
    bootstrap_diffs_b = np.mean([
        ndcg_D_per_job[job] - ndcg_A_per_job[job] 
        for job in bootstrap_jobs
    ])
    bootstrap_diffs.append(bootstrap_diffs_b)

bootstrap_diffs = np.array(bootstrap_diffs)

# Bước 4: Tính 95% CI (Percentile method)
ci_lower = np.percentile(bootstrap_diffs, 2.5)
ci_upper = np.percentile(bootstrap_diffs, 97.5)

print(f"\n95% CI of difference: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Bước 5: P-value (two-tailed)
# P-value = % bootstrap samples mà |bootstrap_diff| >= |observed_diff|
p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

print(f"P-value (two-tailed): {p_value:.4f}")
print(f"  Interpretation: Significant at α=0.05? {p_value < 0.05}")
```

**Lựa chọn thứ 2** (Paired Permutation Test — nếu muốn alternative):

```python
# Permutation Test (under null hypothesis: no difference)

n_permutations = 10000
permutation_diffs = []

for p in range(n_permutations):
    # Dưới null hypothesis, A và D là tương đương
    # → shuffle: với mỗi job, random chọn "difference từ A" hoặc "difference từ D"
    
    signs = np.random.choice([-1, 1], size=len(gold_queries))
    # sign = 1 → difference = (D - A)
    # sign = -1 → difference = (A - D) = -(D - A)
    
    permutation_diff = np.mean([
        signs[i] * (ndcg_D_per_job[job] - ndcg_A_per_job[job])
        for i, job in enumerate(gold_queries)
    ])
    permutation_diffs.append(permutation_diff)

permutation_diffs = np.array(permutation_diffs)

# P-value = % permutations mà |permutation_diff| >= |observed_diff|
p_value_perm = np.mean(np.abs(permutation_diffs) >= np.abs(observed_diff))

print(f"P-value (permutation test): {p_value_perm:.4f}")
```

### 📝 Kết Luận Sửa

- **Xóa mô tả mơ hồ** ở Mục 9.
- **Thêm code Python cụ thể** cho Bootstrap Paired Test (Option 1, recommended).
- **Ghi rõ**: "Resample over jobs (với replacement), không resample pairs".
- **Thêm ghi chú**: "Nếu p-value > 0.05, không thể nói D tốt hơn A với confidence 95% → cần Gold set lớn hơn hoặc dữ liệu train lớn hơn".

---

## 🎯 Tóm Tắt: 5 Sửa Chính

| # | Phần | Lỗi | Sửa |
|----|------|-----|-----|
| 1 | Mục 3–6 | Fleiss' kappa, Kendall's tau_b nhầm lẫn | Precision/Recall S_i trên dev-gold; Cohen's kappa cho annotators |
| 2 | Mục 4 | "Hard negative" mơ hồ | Code với threshold số: 0.85, 0.4, 0.5 |
| 3 | Mục 5 | Pair sampling công thức không rõ | Pseudocode: sample từ Cartesian, limit k_max per job |
| 4 | Mục 7.1–7.2 | Training details thiếu | Hyperparameter table + training loop + early stopping |
| 5 | Mục 9 | Bootstrap/permutation không rõ | Bootstrap Paired Test code cụ thể, resample jobs |

