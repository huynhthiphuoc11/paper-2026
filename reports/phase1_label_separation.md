# BÁO CÁO PHÂN TÁCH LUỒNG NHÃN & RÒ RỈ DỮ LIỆU (PHASE 1 LABEL SEPARATION & LEAKAGE AUDIT REPORT)

**Dự án:** Leakage-Aware Weakly Supervised Learning-to-Rank for Vietnamese CV–Job Matching  
**Pha:** Pha 1 — Leakage Removal & Split Freezing  
**Trạng thái:** HOÀN THÀNH (PASS)  

---

## 1. SƠ ĐỒ PHÂN TÁCH KIẾN TRÚC MỚI (NEW PIPELINE PATH ARCHITECTURE)

Trong Pha 1, đường truyền dữ liệu đã được tái cấu trúc triệt để theo đúng nguyên tắc **Phân tách Tuyệt đối giữa Feature Path và Test Target Path**:

```text
                             RAW CV + RAW JD
                                    │
               ┌────────────────────┴────────────────────┐
               │                                         │
               ▼                                         ▼
      TRAIN / DEV PATH                               TEST PATH
       (Job-Disjoint)                             (Job-Disjoint)
               │                                         │
               ▼                                         ▼
     FEATURE EXTRACTION                         FEATURE EXTRACTION
 (loc_match, skill_iou, exp_score,          (loc_match, skill_iou, exp_score,
  role_match, desc_sem_sim)                  role_match, desc_sem_sim)
               │                                         │
               ▼                                         ▼
   5 ASPECT LABELING FUNCTIONS                   MODEL INFERENCE
       (Fit on Train Only)               (H, A, B, B+, C, D, D+ predictions)
               │                                         │
               ▼                                         │
    PROBABILISTIC AGGREGATION                            │
            (y_prob)                                     │
               │                                         │
               ▼                                         │
        MODEL TRAINING                                   │
 (Learned Weights & Soft-RankNet)                        │
               │                                         │
               └────────────────────┬────────────────────┘
                                    ▼
                         INDEPENDENT EVALUATION
                   (Evaluated on Unseen Test Jobs)
```

---

## 2. MINH CHỨNG TRIỆT XOÁ BÃY LOGIC (PROOF OF NO FEATURE-TO-TEST-TARGET PATH)

1. **New Train Label Path:** `RAW CV-JD Features` $\rightarrow$ `5 Aspect LFs` $\rightarrow$ `Dawid-Skene Aggregation` $\rightarrow$ `y_prob` $\rightarrow$ `Model Training`.
2. **New Test Path:** `Raw Test Pair` $\rightarrow$ `Feature Extraction` $\rightarrow$ `Model Inference` $\rightarrow$ `Predicted Ranking Scores`.
3. **No Feature-to-Test-Target Path:** **Không tồn tại bất kỳ tuyến nào từ `FEATURE_COLS` hoặc `legacy_composite_quality` để tự sinh nhãn kiểm thử cho mô hình.**

---

## 3. PHÂN TÍCH ĐÁNH GIÁ THỰC NGHIỆM VỀ NGUYÊN NHÂN OVERFITTING / UNDERFITTING TRÊN BASELINE GỐC

Đối chiếu bảng kết quả Baseline Pha 0 đã đóng băng:

```text
TABLE 1: MODEL PERFORMANCE ON HISTORICAL LEGACY GOLD SET
                     Model   nDCG@5  nDCG@10   MAP@10      MRR
             H (Heuristic) 0.886662 0.821064 0.192772 0.933333
          A (Baseline BCE) 0.886662 0.821064 0.192772 0.933333
           B (Learned BCE) 0.860358 0.806329 0.183942 0.933333
             B+ (Soft BCE) 1.000000 0.988912 0.310537 1.000000
         C (Fixed RankNet) 0.076937 0.118030 0.012208 0.121873
          D (Main RankNet) 0.057146 0.191942 0.019752 0.126967
D+ (Proposed Soft-RankNet) 1.000000 0.988912 0.310240 1.000000
```

### Phán đoán về B+ và D+ đạt kết quả gần như tuyệt đối ($nDCG@10 \approx 0.989$):
- **Phù hợp với giả thuyết về Rò rỉ Nhãn (Consistent with Label-Construction Leakage):** Mức nDCG gần 1.0 của $B^+$ và $D^+$ dưới quy trình kiểm thử cũ hoàn toàn nhất quán với nguy cơ rò rỉ nhãn đã được xác định, do mô hình và nhãn test cũ cùng khai thác các công thức đặc trưng tuyến tính tương đồng.

### Phán đoán về Mô hình C và D có kết quả thấp ($nDCG@10 \approx 0.11 - 0.19$):
- **Giả thuyết về Nhãn Cứng và Tối ưu hóa:** Điểm số thấp của $C$ và $D$ có thể liên quan đến nhiễu trong nhãn phân ngưỡng cứng ($0.45$), mất cân bằng lớp, hoặc đặc thụ của thuật toán tối ưu hóa Pairwise RankNet khi xử lý các cặp ứng viên nhiễu.

---

## 4. KẾT LUẬN PHA 1

Pha 1 đã hoàn thành đầy đủ 3 tiêu chuẩn nghiệm thu:
1. Đã tạo và validate 100% file đóng băng split: [`data/splits/split_manifest.csv`](file:///d:/NCKH/paper-2026/data/splits/split_manifest.csv).
2. Đã xuất file báo cáo nghiệm thu split: [`reports/phase1_split_validation.json`](file:///d:/NCKH/paper-2026/reports/phase1_split_validation.json).
3. Đã refactor `src/data_loader.py` chuyển nhãn cũ thành `legacy_composite_quality` và `legacy_gold_relevance`, cắt bỏ tuyến sinh nhãn kiểm thử circular.

Đủ điều kiện để chuyển sang **Pha 2: LF Design, ABSTAIN Policy, LF Dependency Audit & Probabilistic Label Aggregation**.
