# Hướng Đi Đúng cho Bài Toán CV–Job Matching Tiếng Việt
## (Bản cũ — SUPERSEDED)

> **SUPERSEDED bởi [`DE_CUONG_CHOT.md`](DE_CUONG_CHOT.md).** Không dùng làm nguồn đề cương/paper. Giữ chỉ để truy vết lịch sử.

---

## 1. Bối Cảnh & Động Lực

Bài báo nền (baseline): Huynh et al. 2025, *"A Vietnamese Job-Candidate Matching System Based on Embedding Models and Gradient Boosting"* (DEFI 2025).

Baseline dùng:
- Công thức trọng số cố định, chọn tay: $Score = 0.30 S_{loc} + 0.25 S_{skill} + 0.20 S_{exp} + 0.15 S_{role} + 0.10 S_{desc}$
- Ngưỡng nhị phân cố định (0.45) để gán nhãn match/non-match
- Train/test split lấy từ **cùng công thức** dùng để tạo nhãn
- Dữ liệu nhỏ: 80 jobs, 117 candidates, 5.817 cặp

**Vấn đề phương pháp luận** — gọi là *label-source-dependent evaluation* (không phải "data leakage" theo nghĩa thông thường):
Khi nhãn train và nhãn test cùng sinh ra từ một công thức, mọi chỉ số (F1, AUC…) chỉ đo được "mô hình khớp công thức tốt đến đâu", không đo được "công thức đó có đúng thật không".

---

## 2. Câu Hỏi Nghiên Cứu (một câu hỏi chính)

**RQ:** Khi đánh giá trên một tập gold độc lập (không sinh từ công thức cũ, không dùng để train), việc (i) **học trọng số** kết hợp thay vì cố định tay, và (ii) dùng mục tiêu **xếp hạng (ranking)** thay vì phân loại nhị phân, có cải thiện chất lượng xếp hạng CV–Job hay không — và **đóng góp của từng thay đổi** là bao nhiêu?

Hai trục của ablation (sub-questions) trong cùng một RQ:
- **Trục trọng số:** fixed (công thức cũ) vs. learned (từ dữ liệu)
- **Trục mục tiêu:** binary classification (BCE) vs. xếp hạng cặp (Pairwise RankNet)

---

## 3. Ba Nguồn Tín Hiệu Yếu (Weak Signals) — dùng để **tạo nhãn train**

| # | Tín hiệu | Cách tính | Ngưỡng |
|---|----------|-----------|--------|
| S1 | **Semantic retrieval** | Cosine similarity giữa embedding (title+description) của CV và Job, dùng embedding tiếng Việt (PhoBERT / multilingual-e5) | θ_s ≈ 0.5, hiệu chỉnh trên dev-gold |
| S2 | **Skill ontology matching** | Chuẩn hoá kỹ năng qua ontology (alias mapping), tính overlap có trọng số: $0.7 \cdot \text{overlap}_{required} + 0.3 \cdot \text{overlap}_{nice}$ | θ_sk ≈ 0.6–0.7 |
| S3 | **Experience compatibility** | So khớp số năm kinh nghiệm CV với khoảng yêu cầu của Job, phạt tuyến tính khi thiếu/dôi | θ_e ≈ 0.7 |

### 3.1. Kiểm Chứng Chất Lượng Từng Signal (bắt buộc trước khi train chính)

**Dev-gold set nhỏ** (tách biệt hoàn toàn với Gold set cuối ở mục 6):
- Lấy mẫu ngẫu nhiên 500–1000 cặp CV-Job, người chú thích tay gán match/non-match
- Dùng làm ground-truth để đo **độ chính xác từng S_i**

```python
def evaluate_signal(signal_pred, gold_labels):
    """
    signal_pred[i]: True/False (signal nói match hay non-match)
    gold_labels[i]: 1/0 (người chú thích nói match hay non-match)
    """
    TP = sum(p and g for p, g in zip(signal_pred, gold_labels))
    FP = sum(p and not g for p, g in zip(signal_pred, gold_labels))
    FN = sum((not p) and g for p, g in zip(signal_pred, gold_labels))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1}
```

**Mục tiêu**: Precision ≥ 0.85 cho từng signal. Nếu không đạt: điều chỉnh ngưỡng (θ_s, θ_sk, θ_e) hoặc cách tính signal, kiểm tra lại.

**Không dùng dev-gold set này sau đó** — tách biệt hoàn toàn khỏi training và evaluation chính.

### 3.2. Mức Độ Phụ Thuộc Giữa 3 Signal (câu hỏi khác, tách biệt với 3.1)

Mục 3.1 đo *độ chính xác* của từng S_i. Câu hỏi khác, độc lập: **3 signal có tương quan/phụ thuộc lẫn nhau đến mức nào?** — quan trọng vì nếu gần như trùng hoàn toàn, "3 nguồn heterogeneous" chỉ là hình thức.

Đo bằng **Fleiss' kappa** trên chính output nhị phân (vượt ngưỡng hay không) của 3 signal, tính trên toàn bộ dữ liệu (không cần ground-truth):

```python
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

# binary_votes: shape (n_pairs, 3) — mỗi cột là 1 signal, giá trị 0/1
table, _ = aggregate_raters(binary_votes)
kappa = fleiss_kappa(table)
print(f"Fleiss' kappa giữa 3 weak signals: {kappa:.4f}")
```

Đây là **bằng chứng mô tả** (descriptive), không phải tiêu chí pass/fail. Ghi rõ trong paper: "3 signal có kappa = X, cho thấy [mức độ tương quan]".

**Quan trọng**: Đây là Fleiss' kappa cho **3 weak signals**. Khác biệt hoàn toàn với Cohen's kappa ở mục 6 (cho người chú thích trên Gold set). Hai phép đo này phục vụ hai câu hỏi riêng biệt, không gộp chung.

---

## 4. Gán Nhãn: Vote-Based + Ngưỡng Cụ Thể

Với vote = số tín hiệu vượt ngưỡng ∈ {0,1,2,3}:

```python
THETA_UPPER = 0.85   # signal "rất cao"
THETA_LOWER = 0.4    # signal "rất thấp"
DELTA = 0.5           # chênh lệch gap tối thiểu

def get_label(S1, S2, S3, theta_s=0.5, theta_sk=0.6, theta_e=0.7):
    """
    Trả về (label, label_type):
      label ∈ {0, 1, 2, None}
      label_type ∈ {'relevant', 'clear_non_match', 'hard_negative',
                     'soft_ambiguous', 'uncertain'}
    """
    votes = sum([S1 > theta_s, S2 > theta_sk, S3 > theta_e])
    signals = [S1, S2, S3]
    max_s, min_s = max(signals), min(signals)

    # Positive: tất cả 3 đồng ý match
    if votes == 3:
        return 2, "relevant"
    
    # Negative rõ ràng: tất cả 3 đồng ý non-match
    elif votes == 0:
        return 0, "clear_non_match"
    
    # Ambiguous (votes == 1 or 2): cần kiểm tra thêm
    else:
        # Mismatch mạnh: một signal rất cao, một signal rất thấp, gap >= 0.5
        if max_s >= THETA_UPPER and min_s < THETA_LOWER and (max_s - min_s) >= DELTA:
            return 0, "hard_negative"
        
        # Tất cả signals gần ngưỡng, không có tín hiệu rõ ràng
        elif all(THETA_LOWER <= si <= THETA_UPPER for si in signals):
            return None, "uncertain"   # Loại khỏi train
        
        # Các trường hợp còn lại: có thông tin nhưng không hẳn rõ ràng
        else:
            return 1, "soft_ambiguous"  # Dùng train với confidence thấp
```

### Bảng Quyết Định Chi Tiết

| Vote | max_s | min_s | gap = max - min | Kết luận | Label | Dùng train? |
|------|-------|-------|-----------------|----------|-------|------------|
| 3 | - | - | - | Cả 3 đồng ý match | 2 | ✓ positive |
| 0 | - | - | - | Cả 3 đồng ý non-match | 0 | ✓ strong negative |
| 1–2 | ≥0.85 | <0.4 | ≥0.5 | Mismatch mạnh | 0 | ✓ hard negative |
| 1–2 | ∈[0.4,0.85] | ∈[0.4,0.85] | - | Tất cả gần ngưỡng | ∅ | ✗ **loại khỏi train** |
| 1–2 | ngoài | ngoài | <0.5 | Chưa rõ nhưng có thông tin | 1 | ✓ soft ambiguous |

---

## 5. Sinh Cặp Huấn Luyện (Pairwise LTR)

### 5.1. Sampling Có Kiểm Soát

```python
import random

def create_training_pairs(labeled_data, max_pairs_per_job=100):
    """
    labeled_data: dict[(cv, job)] = label ∈ {0, 1, 2}
    Trả về list of pairs cho pairwise RankNet
    """
    train_pairs = []
    jobs = set(job for (_, job) in labeled_data.keys())

    for job in jobs:
        relevant = [cv for (cv, j) in labeled_data 
                    if j == job and labeled_data[(cv, j)] == 2]
        hard_neg = [cv for (cv, j) in labeled_data 
                    if j == job and labeled_data[(cv, j)] == 0]
        soft_ambig = [cv for (cv, j) in labeled_data 
                      if j == job and labeled_data[(cv, j)] == 1]

        if not relevant or not hard_neg:
            continue  # Job không đủ dữ liệu, bỏ qua

        # Tất cả candidates negative (hard + soft_ambig)
        candidates_neg = hard_neg + soft_ambig
        
        # Lấy mẫu cố định K = min(tất cả pairs, max_pairs_per_job)
        all_pairs = [(cv_p, cv_n) for cv_p in relevant for cv_n in candidates_neg]
        K = min(len(all_pairs), max_pairs_per_job)
        sampled = random.sample(all_pairs, K)

        for cv_pos, cv_neg in sampled:
            train_pairs.append({
                'job': job,
                'doc_1': cv_pos,      # should rank higher
                'doc_2': cv_neg,      # should rank lower
                'label': 1            # doc_1 > doc_2
            })

    return train_pairs

# Kiểm tra phân bố
pairs_per_job = {}
for pair in train_pairs:
    j = pair['job']
    pairs_per_job[j] = pairs_per_job.get(j, 0) + 1

ratio = max(pairs_per_job.values()) / min(pairs_per_job.values())
print(f"Max/min pairs per job ratio: {ratio:.2f} (target: < 3)")
```

### 5.2. ⚠️ Cảnh Báo: Aggregation Function

Nếu áp dụng hard-negative mining bằng margin:

```python
# ❌ SAI: dùng lại công thức 5-trọng-số gốc của baseline
margin = 0.30*S_loc[p] + 0.25*S_skill[p] + ... - (0.30*S_loc[n] + ...)

# ✓ ĐÚNG: trung bình đơn giản của 3 weak signals
margin = (S1[p] + S2[p] + S3[p]) / 3 - (S1[n] + S2[n] + S3[n]) / 3
```

**Quy tắc bắt buộc**: Bất kỳ bước nào trong pipeline train — kể cả chỉ để filter/mining — **không được dùng lại công thức 5-trọng-số gốc**.

---

## 6. Gold Set Đánh Giá — Job-Disjoint & Tách Biệt Hoàn Toàn

- **Job-disjoint**: các job trong Gold set hoàn toàn không xuất hiện trong dữ liệu dùng sinh weak-label hay train.
- **Quy mô**: 10–15 jobs (pilot scale, khai báo rõ không mở rộng giả), mỗi job 10–20 CV.
- **Chú thích**: người chú thích tay gán nhãn nhị phân `relevant / irrelevant` cho mỗi (CV, Job).
- **Đo đồng thuận giữa người chú thích**: dùng **Cohen's kappa** (2 annotators) hoặc **Fleiss' kappa** (≥3 annotators).

  $$\kappa = \frac{p_o - p_e}{1 - p_e}$$

  với $p_o$ = observed agreement, $p_e$ = expected by chance. Mục tiêu: κ ≥ 0.7.

  > **Lưu ý quan trọng**: Đây là Cohen's/Fleiss' kappa cho **người chú thích trên Gold set** (mục 6) — riêng biệt hoàn toàn với Fleiss' kappa ở mục 3.2 (đo phụ thuộc giữa 3 weak signals). Hai phép đo này là hai con số riêng biệt, phục vụ hai câu hỏi khác nhau. **Không gộp chung**.

- **Không đụng tới cho tới bước đánh giá cuối** — không tune hyperparameter, không xem trước.

---

## 7. Cơ Chế Học Trọng Số & Cơ Chế Xếp Hạng (Chi Tiết Kỹ Thuật)

### 7.1. Trục Trọng Số: "Fixed" vs. "Learned"

**Fixed** (H, A, B): $score = w_1 S_1 + w_2 S_2 + w_3 S_3$ với $w$ cố định tay (ví dụ $w_1=w_2=w_3=1/3$), không đổi.

**Learned** (C, D): input $x = [S_1, S_2, S_3]$, mô hình học hàm $f_\theta(x)$:

```python
import torch
import torch.nn as nn

# Phương án đơn giản: Linear Ranker
class LinearRanker(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        """x: shape (batch, 3) — [S1, S2, S3]"""
        return self.linear(x).squeeze(-1)  # shape (batch,)

# Phương án khuyến nghị: MLP nhỏ (bắt tương tác phi tuyến)
class RankingMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """x: shape (batch, 3)"""
        h = self.relu(self.fc1(x))      # shape (batch, hidden_dim)
        return self.fc2(h).squeeze(-1)  # shape (batch,)
```

Khác biệt cốt lõi: tham số (trọng số/MLP weights) là **tham số học từ train set qua gradient descent**, không phải số người thiết kế tự chọn.

### 7.2. Trục Mục Tiêu: Classification vs. Ranking

**Classification** (A, C):
```python
import torch.nn.functional as F

# Loss: Binary Cross-Entropy with Logits
logit = model(x)  # raw output, chưa qua sigmoid
y = torch.ones_like(logit)  # label = 1 (match) cho doc_pos
loss = F.binary_cross_entropy_with_logits(logit, y)
```

Tại sao dùng `binary_cross_entropy_with_logits` thay vì tự viết `log(sigmoid(...))`?
→ PyTorch tối ưu hóa số học và xử lý gradient ổn định, tránh numerical underflow/overflow.

**Ranking / RankNet** (B, D):
```python
# Loss: Pairwise RankNet
s_i = model(x_i)  # score cho cv_i (should rank higher)
s_j = model(x_j)  # score cho cv_j (should rank lower)

diff = s_i - s_j
# P_ij = sigmoid(diff) = xác suất cv_i rank cao hơn cv_j
# L = -log(P_ij) = log(1 + exp(-diff))

loss = F.softplus(-diff)  # numerically stable implementation
```

Tại sao dùng `softplus(-diff)` thay vì tự viết?
→ `softplus(x) = log(1 + exp(x))` được implement an toàn số học, tránh numerical issues.

**Quan trọng**: Không dùng numpy trên tensor PyTorch có `requires_grad=True`:
```python
# ❌ SAI: numpy không nằm trong autograd graph
loss = np.log(1 + np.exp(-diff))  # backward() sẽ bị lỗi hoặc không gradient

# ✓ ĐÚNG: torch functions giữ autograd chain
loss = F.softplus(-diff)  # backward() hoạt động bình thường
```

Khi suy luận (inference): chỉ cần $s = f_\theta(x)$ cho từng CV, sort giảm dần, lấy top-K — không cần ngưỡng 0.5.

---

## 8. Thiết Kế Ablation — Bảng Factorial 2×2 + Baseline

| Điều kiện | Trọng số | Mục tiêu | Vai trò |
|-----------|----------|----------|---------|
| **H** | Fixed | Không train — sort trực tiếp | Mốc tham chiếu thuần heuristic |
| **A** | Fixed | BCE classification | Tái tạo cách làm của baseline paper |
| **B** | Fixed | Pairwise RankNet | Cô lập hiệu ứng đổi mục tiêu (giữ trọng số cố định) |
| **C** *(optional)* | Learned | BCE classification | Cô lập hiệu ứng học trọng số (giữ mục tiêu classification) |
| **D** (chính) | Learned | Pairwise RankNet | Kết hợp cả hai cải tiến |

Chuỗi so sánh chính: **H → A → B → D**.

---

## 9. Chi Tiết Huấn Luyện

### 9.1. Data Split

```
Weak-labeled data (từ 3 signals, sau khi loại uncertain & soft_ambiguous)
├── Train: 80% — cập nhật tham số qua gradient descent
└── Validation: 20% — tune hyperparameter, early stopping
                        (KHÔNG dùng Gold set ở đây!)
```

### 9.2. Hyperparameter

| Hyperparameter | Recommended | Search Range |
|---|---|---|
| Learning rate | 0.001 | [0.0001, 0.01] |
| Batch size | 32 | [16, 32, 64] |
| L2 regularization | 0.001 | [0, 0.001, 0.01] |
| Hidden dim (MLP) | 64 | [32, 64, 128] |
| Max epochs | 100 | + early stopping |
| Early-stopping patience | 10 | epochs |

### 9.3. Training Loop (Dùng PyTorch — an toàn autograd)

```python
import torch
import torch.optim as optim
import torch.nn.functional as F

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
best_val_loss = float('inf')
patience_counter = 0
max_patience = 10

for epoch in range(1, 101):
    # ===== TRAINING =====
    model.train()
    train_loss_total = 0.0
    n_batches = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        x_i, x_j = batch['x_i'], batch['x_j']  # Tensor shape (batch, 3)
        s_i = model(x_i)  # shape (batch,)
        s_j = model(x_j)  # shape (batch,)

        if condition in ['B', 'D']:
            # Pairwise RankNet loss
            loss = F.softplus(-(s_i - s_j)).mean()
        else:
            # Classification (A, C)
            loss = F.binary_cross_entropy_with_logits(s_i, torch.ones_like(s_i)).mean()

        loss.backward()
        optimizer.step()
        train_loss_total += loss.item()
        n_batches += 1

    train_loss_avg = train_loss_total / n_batches

    # ===== VALIDATION =====
    model.eval()
    val_loss_total = 0.0
    n_val_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x_i, x_j = batch['x_i'], batch['x_j']
            s_i = model(x_i)
            s_j = model(x_j)
            
            if condition in ['B', 'D']:
                loss = F.softplus(-(s_i - s_j)).mean()
            else:
                loss = F.binary_cross_entropy_with_logits(s_i, torch.ones_like(s_i)).mean()
            
            val_loss_total += loss.item()
            n_val_batches += 1

    val_loss_avg = val_loss_total / n_val_batches

    # ===== EARLY STOPPING =====
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
        print(f"Epoch {epoch}: train={train_loss_avg:.4f}, val={val_loss_avg:.4f} [BEST]")
    else:
        patience_counter += 1
        print(f"Epoch {epoch}: train={train_loss_avg:.4f}, val={val_loss_avg:.4f}")
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Load best model cho evaluation
model.load_state_dict(torch.load("best_model.pt"))
```

---

## 10. Đánh Giá Trên Gold Set

### 10.1. Ranking Metrics

```python
from sklearn.metrics import ndcg_score
import numpy as np

def mrr_score(y_true, y_pred):
    """sklearn không có MRR — tự viết."""
    ranked_indices = np.argsort(y_pred)[::-1]
    for rank, idx in enumerate(ranked_indices, start=1):
        if y_true[idx] == 1:
            return 1.0 / rank
    return 0.0

# Tính metrics cho từng condition
for condition in ['H', 'A', 'B', 'C', 'D']:
    ndcg_5_scores, ndcg_10_scores, mrr_scores = [], [], []
    
    for job in gold_jobs:
        y_true = gold_labels[job]  # binary labels
        y_pred = predict_scores(condition, job)  # continuous scores
        
        ndcg_5 = ndcg_score([y_true], [y_pred], k=5)
        ndcg_10 = ndcg_score([y_true], [y_pred], k=10)
        mrr = mrr_score(y_true, y_pred)
        
        ndcg_5_scores.append(ndcg_5)
        ndcg_10_scores.append(ndcg_10)
        mrr_scores.append(mrr)

    print(f"Condition {condition}:")
    print(f"  NDCG@5:  {np.mean(ndcg_5_scores):.4f} ± {np.std(ndcg_5_scores):.4f}")
    print(f"  NDCG@10: {np.mean(ndcg_10_scores):.4f} ± {np.std(ndcg_10_scores):.4f}")
    print(f"  MRR:     {np.mean(mrr_scores):.4f} ± {np.std(mrr_scores):.4f}")
```

### 10.2. Statistical Significance — Bootstrap Paired Test

Vì Gold set nhỏ (~10-15 jobs), **không dùng t-test tham số** (giả định normal distribution không đủ để thỏa).
Dùng **Bootstrap Paired Resampling**:

```python
def bootstrap_paired_test(y_true_list, y_pred_base_list, y_pred_new_list, 
                          k=5, n_bootstrap=10000, random_seed=42):
    """
    Bootstrap test giữa hai methods.
    
    Args:
      y_true_list: list of binary labels per job
      y_pred_base_list, y_pred_new_list: list of scores per job
      k: ngưỡng cho NDCG@k
      n_bootstrap: số lần resample
    
    Returns:
      dict với observed diff, 95% CI, p-value
    """
    np.random.seed(random_seed)
    
    # Bước 1: Tính metric cho từng job
    metrics_base = [ndcg_score([yt], [yp], k=k) 
                    for yt, yp in zip(y_true_list, y_pred_base_list)]
    metrics_new = [ndcg_score([yt], [yp], k=k) 
                   for yt, yp in zip(y_true_list, y_pred_new_list)]
    
    # Bước 2: Tính chênh lệch quan sát
    diffs = np.array(metrics_new) - np.array(metrics_base)
    observed_mean_diff = np.mean(diffs)
    
    # Bước 3: Bootstrap resample jobs (WITH replacement)
    bootstrap_mean_diffs = []
    n_jobs = len(diffs)
    
    for b in range(n_bootstrap):
        indices = np.random.choice(n_jobs, size=n_jobs, replace=True)
        bootstrap_mean = np.mean(diffs[indices])
        bootstrap_mean_diffs.append(bootstrap_mean)
    
    bootstrap_mean_diffs = np.array(bootstrap_mean_diffs)
    
    # Bước 4: Tính 95% CI (Percentile method)
    ci_lower = np.percentile(bootstrap_mean_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_mean_diffs, 97.5)
    
    # Bước 5: Two-tailed p-value
    p_value = np.mean(np.abs(bootstrap_mean_diffs) >= np.abs(observed_mean_diff))
    
    return {
        'observed_diff': observed_mean_diff,
        'ci_95': (ci_lower, ci_upper),
        'p_value': p_value,
        'significant_at_05': p_value < 0.05
    }

# Sử dụng
result_D_vs_A = bootstrap_paired_test(
    gold_y_true, gold_y_pred_A, gold_y_pred_D, k=5, n_bootstrap=10000
)

print(f"D vs A (NDCG@5):")
print(f"  Mean diff: {result_D_vs_A['observed_diff']:.4f}")
print(f"  95% CI: [{result_D_vs_A['ci_95'][0]:.4f}, {result_D_vs_A['ci_95'][1]:.4f}]")
print(f"  P-value: {result_D_vs_A['p_value']:.4f}")
print(f"  Significant? {result_D_vs_A['significant_at_05']}")
```

---

## 11. Error Analysis

Với mỗi cặp so sánh (D vs. A, D vs. H):

```python
def error_analysis(gold_jobs, y_scores_dict, y_true_dict, job_metadata):
    """
    y_scores_dict[condition] = [scores per job]
    y_true_dict[job] = binary labels
    job_metadata[job] = {'level': 'junior'|'mid'|'senior', 'domain': 'IT'|...}
    """
    errors = []
    
    for job in gold_jobs:
        y_true = y_true_dict[job]
        ndcg_D = ndcg_score([y_true], [y_scores_dict['D'][job]], k=5)
        ndcg_A = ndcg_score([y_true], [y_scores_dict['A'][job]], k=5)
        
        if (ndcg_A - ndcg_D) >= 0.05:  # D tệ hơn A ≥ 5%
            errors.append({
                'job': job,
                'ndcg_D': ndcg_D,
                'ndcg_A': ndcg_A,
                'drop': ndcg_A - ndcg_D,
                'level': job_metadata[job]['level'],
                'domain': job_metadata[job]['domain']
            })
    
    # Phân tích theo level (chỉ những levels thực sự có data)
    for level in set(e['level'] for e in errors):
        level_errs = [e for e in errors if e['level'] == level]
        print(f"\nLevel '{level}': {len(level_errs)} errors (drop ≥ 5%)")
        if level_errs:
            print(f"  Mean drop: {np.mean([e['drop'] for e in level_errs]):.4f}")
    
    # Phân tích theo domain (chỉ những domains thực sự có data)
    for domain in set(e['domain'] for e in errors):
        domain_errs = [e for e in errors if e['domain'] == domain]
        print(f"\nDomain '{domain}': {len(domain_errs)} errors")
        if domain_errs:
            print(f"  Mean drop: {np.mean([e['drop'] for e in domain_errs]):.4f}")
```

---

## 12. Hạn Chế (Ghi Rõ, Không Né Tránh)

1. **Gold set nhỏ** (10–15 jobs, pilot scale)
   - → Không khẳng định tổng quát hoá toàn thị trường lao động Việt Nam
   - → Chỉ dùng để so sánh các phương pháp, không nói chắc hiệu năng tuyệt đối

2. **Weak labels có sai số** (~80–85% precision theo dev-gold)
   - → Nhiễu này lan vào nhãn train, dù đã lọc uncertain
   - → Model có thể học được noise thay vì signal

3. **Skill Ontology không hoàn chỉnh**
   - → Kỹ năng mới, viết tắt không chuẩn trong CV tiếng Việt không bắt được
   - → S2 có thể miss match legit

4. **Đánh giá hoàn toàn offline**
   - → Không đo được mức hài lòng thực tế của nhà tuyển dụng
   - → Ranking metrics (NDCG) có thể không capture được user satisfaction

5. **Domain của Gold set có thể lệch**
   - → Nếu CV từ một nguồn, kết quả có thể không tổng quát sang nguồn khác
   - → Cần test cross-domain (future work)

---

## 13. Danh Sách Kiểm Tra (Self-Check Trước Khi Implement)

- [ ] Không đâu trong pipeline train dùng công thức 5-trọng-số gốc (kể cả mining/lọc)
- [ ] Dev-gold (mục 3.1, 500–1000 cặp) tách biệt hoàn toàn khỏi Gold set (mục 6, 10–15 jobs)
- [ ] Hard-negative = mismatch mạnh (max≥0.85, min<0.4, gap≥0.5), KHÔNG phải "gần ngưỡng"
- [ ] Uncertain (mọi signal ∈ [0.4, 0.85]) bị loại khỏi train; soft-ambiguous là nhóm khác
- [ ] Gold set job-disjoint hoàn toàn với train/weak-label generation
- [ ] Fleiss' kappa (3 signals, mục 3.2) là một con số riêng biệt, **tách biệt hoàn toàn** từ Cohen's kappa (annotators, mục 6)
- [ ] Ablation dùng đúng 5 điều kiện H/A/B/C/D như bảng mục 8
- [ ] Validation set từ weak-labeled data (80/20), KHÔNG dùng Gold set để tune
- [ ] Significance test dùng **Bootstrap Paired**, KHÔNG t-test tham số
- [ ] Training loop dùng `torch` functions (F.softplus, F.binary_cross_entropy_with_logits), KHÔNG numpy trên tensor autograd

---

## 14. Timeline (Ước Lượng)

| Giai đoạn | Task | Thời gian |
|-----------|------|-----------|
| Chuẩn bị | Skill Ontology, tính 3 signals, dev-gold + kiểm chứng ngưỡng | 4–6 tuần |
| Sinh dữ liệu train | Gán nhãn (votes → label), sinh training pairs | 2–3 tuần |
| Gold set | Chú thích tay 10–15 job, tính Cohen's/Fleiss' kappa | 3–4 tuần |
| Huấn luyện | Implement H/A/B/C/D, tune hyperparameter, train | 4–6 tuần |
| Đánh giá | NDCG/MRR, Bootstrap test, error analysis | 2–3 tuần |
| Viết báo | Tổng hợp kết quả, viết paper | 2–3 tuần |

**Tổng cộng**: ~17–25 tuần (~4–6 tháng)

---

## 15. Kết Luận: Hướng Đi Đúng

| Khía cạnh | ❌ Baseline (Sai) | ✓ Đề xuất (Đúng) |
|-----------|-----------------|-----------------|
| Vấn đề | Công thức cố định, không kiểm chứng | Tách nguồn tạo nhãn & nguồn đánh giá |
| Nhãn train | Công thức cũ → vòng lặp circularity | Ba weak signals (S1, S2, S3), loại uncertain |
| Bài toán | Binary classification (BCE) | Learning-to-Rank (Pairwise RankNet) |
| Metrics | F1, Precision, Recall | NDCG@K, MRR |
| Đánh giá | Test set từ công thức cũ | Gold set chú thích tay, job-disjoint |
| Kết luận | "Model khớp công thức tốt" (vô nghĩa) | "Model xếp hạng tốt hơn công thức" (thực tế) |

**Lợi Ích Cụ Thể**:
- ✓ Thoát khỏi vòng lặp học lại công thức cũ
- ✓ Đánh giá trên thứ mà người dùng thực sự quan tâm (xếp hạng, top-K)
- ✓ Có thể phát hiện khi công thức cũ sai
- ✓ Phương pháp luận sạch sẽ, reproducible, không lẩn tránh hạn chế

---

## Tài Liệu Tham Khảo (Future Reference)

Thêm vào paper:
- Learning-to-Rank overview: Cao et al. "Learning to Rank: From Pairwise Approach to Listwise Approach"
- RankNet: Burges et al. "Learning to Rank using Gradient Descent"
- Bootstrap test: Efron & Tibshirani "An Introduction to the Bootstrap"
- Vietnamese embeddings: PhoBERT paper, multilingual-e5 docs
