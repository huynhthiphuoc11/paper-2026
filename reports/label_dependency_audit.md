# BÁO CÁO AUDIT TÍNH PHỤ THUỘC NHÃN & VÒNG LẶP ĐÁNH GIÁ (LABEL DEPENDENCY & CIRCULAR EVALUATION AUDIT REPORT)

**Dự án:** Vietnamese CV-Job Ranking Benchmark (`d:\NCKH\paper-2026`)  
**Pha:** Pha 0 — Freeze Baseline & Code Audit  
**Ngày thực hiện:** 21/08/2026  

---

## 1. MẪU BẢNG AUDIT TRÍCH XUẤT TRƯỜNG DỮ LIỆU (FIELD AUDIT TABLE)

| Tên Trường (Field Name) | Nguồn Gốc (Source) | Loại (Type) | Mục Đích Sử Dụng | Trạng Thái Sau Audit |
| :--- | :--- | :--- | :--- | :--- |
| `loc_match` | RAW CV & JD Address | Feature | Input Feature cho tất cả các mô hình | **GIỮ NGUYÊN** trong Feature Set |
| `skill_iou` | RAW Skills overlap | Feature | Input Feature cho tất cả các mô hình | **GIỮ NGUYÊN** trong Feature Set |
| `exp_score` | RAW Experience years | Feature | Input Feature cho tất cả các mô hình | **GIỮ NGUYÊN** trong Feature Set |
| `role_match` | RAW Title TF-IDF | Feature | Input Feature cho tất cả các mô hình | **GIỮ NGUYÊN** trong Feature Set |
| `desc_sem_sim` | RAW Description TF-IDF| Feature | Input Feature cho tất cả các mô hình | **GIỮ NGUYÊN** trong Feature Set |
| `heuristic_score` | Công thức tổng hợp | Derived Feature | Input cho Baseline Model H, A, C | **GIỮ NGUYÊN** làm Baseline |
| `heuristic_label` | Ngưỡng `heuristic_score >= 0.45` | Weak Target | Nhãn nhị phân cho Baseline Model A, B, D | **GIỮ NGUYÊN** làm Weak Label |
| `composite_quality` | Công thức tổng hợp | Derived Target | Nguồn sinh ra `gold_relevance` | ❌ **LOẠI BỎ KHỎI TẬP TEST GOLD** |
| `gold_relevance` | Ngưỡng `composite_quality` | Derived Target | Target cho Table 1 Evaluation | ❌ **LOẠI BỎ KHỎI TẬP TEST GOLD** |

---

## 2. TRACE CÂY PHỤ THUỘC NHÃN (CIRCULAR LABEL DEPENDENCY TREE)

Đã thực hiện trace chính xác từng dòng mã nguồn trong file [`src/data_loader.py`](file:///d:/NCKH/paper-2026/src/data_loader.py):

```text
RAW CV (User Profile) + RAW JD (Job Posting)
                    │
                    ▼  (Dòng 75-96: Trích xuất 5 đặc trưng cơ bản)
 ┌──────────────────┼──────────────────┬──────────────────┬──────────────────┐
 │                  │                  │                  │                  │
 ▼                  ▼                  ▼                  ▼                  ▼
loc_match        skill_iou          exp_score          role_match       desc_sem_sim
 │                  │                  │                  │                  │
 ├──────────────────┴──────────────────┼──────────────────┴──────────────────┤
 │                                     │                                     │
 │ (Dòng 98-103: Heuristic Score)       │ (Dòng 108: Composite Quality)       │
 ▼                                     ▼                                     │
heuristic_score = 0.30*loc_match       composite_quality =                   │
                + 0.25*skill_iou                           0.35*skill_iou    │
                + 0.20*exp_score                         + 0.35*desc_sem_sim │
                + 0.15*role_match                        + 0.15*exp_score    │
                + 0.10*desc_sem_sim                      + 0.15*role_match   │
 │                                     │                                     │
 ▼ (Dòng 105)                          ▼ (Dòng 109-114)                    │
heuristic_label = 1 if >= 0.45 else 0  gold_relevance = 2 if (comp >= 0.40   │
                                                          and loc > 0) else..│
                                       │                                     │
                                       ▼                                     │
                                gold_label_binary = 1 if gold_rel >= 1 else 0│
                                       │                                     │
                                       └──────────────────┬──────────────────┘
                                                          ▼
                                            Circular Test Target (Table 1)
```

---

## 3. TRẢ LỜI 4 CÂU HỎI BẮT BUỘC TRƯỚC KHI SANG PHA 1

### Câu 1: Target hiện tại chính xác được sinh ở dòng nào?
- Target `gold_relevance` được sinh trực tiếp tại **Dòng 108–114** trong file [`src/data_loader.py`](file:///d:/NCKH/paper-2026/src/data_loader.py#L108-L114).
- Nhãn nhị phân `gold_label_binary` được sinh tại **Dòng 129**.

### Câu 2: Những feature nào trực tiếp/gián tiếp tham gia tạo target?
- **4 đặc trưng trực tiếp:** `skill_iou` (trọng số 35%), `desc_sem_sim` (trọng số 35%), `exp_score` (trọng số 15%), `role_match` (trọng số 15%).
- **1 đặc trưng điều kiện:** `loc_match` (điều kiện `loc_match > 0` để gán nhãn mức 2).
- **Kết luận:** 100% bộ đặc trưng đầu vào (`FEATURE_COLS`) đều tham gia vào công thức gán nhãn giả lập `gold_relevance`.

### Câu 3: Test split hiện tại có thực sự Job-disjoint không?
- **Có, nhưng ở mức mã nguồn khởi tạo ngẫu nhiên:** Hàm `CVJobDatasetLoader.get_job_disjoint_splits()` tại Dòng 223–239 thực hiện `rng.shuffle(all_jobs)` với `seed = 42` và chia danh sách Job theo tỷ lệ 80% Train (64 Jobs), 5% Dev (4 Jobs), 15% Test (12 Jobs).
- **Vấn đề:** Do chưa được đóng băng ra file `split_manifest.csv`, nếu thay đổi seed hoặc tham số `n_jobs`, danh sách Job trong Test set sẽ bị xáo trộn.

### Câu 4: Baseline hiện tại có reproduce được với seed = 42 không?
- **Có, 100% tái lập (Reproducible).** Kết quả thực nghiệm kiểm thử sinh ra hoàn toàn khớp:
  - Dataset: 5.600 cặp (4.480 Train, 280 Dev, 840 Test).
  - Metrics Table 1 (Gold Set cũ):
    - Model H (Heuristic): nDCG@10 = 0.821064, MRR = 0.933333
    - Model B (Learned BCE): nDCG@10 = 0.806329, MRR = 0.933333
    - Model B+ (Soft BCE): nDCG@10 = 0.988912, MRR = 1.000000
    - Model D+ (Soft-RankNet): nDCG@10 = 0.988912, MRR = 1.000000
  - Circularity Divergence Index ($D_{\text{circ}}$) = **0.8461** (Xác nhận rank reversal mạnh mẽ giữa Heuristic Test và Gold Test hiện tại).

---

## 4. KẾT LUẬN PHA 0

Pha 0 đã hoàn thành đầy đủ nhiệm vụ **Freeze Baseline & Audit**, không làm thay đổi bất kỳ dòng code logic nào của hệ thống. Đủ điều kiện để nghiệm thu và chuyển sang **Pha 1: Leakage Removal & Split Manifest Creation**.
