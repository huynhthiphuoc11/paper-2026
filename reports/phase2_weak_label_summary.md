# BÁO CÁO KẾT QUẢ PHASE 2.1: LF SANITY CHECK & CONDITIONAL DISTRIBUTION AUDIT

**Dự án:** Leakage-Aware Weakly Supervised Learning-to-Rank for Vietnamese CV–Job Matching  
**Pha:** Phase 2.1 — Sanity Check 5 Yêu Cầu & Kiểm Thử Phân Phối Xác Suất Nhãn Yếu  
**Trạng thái:** PASS CHÍNH THỨC (HOÀN THÀNH 100%)  

---

## 1. KẾT QUẢ KIỂM THỬ 5 YÊU CẦU SANITY CHECK (PHASE 2.1)

### Check A: Kiểm tra chi tiết `LF_exp` & Xử lý Điểm Khối (Point Mass)
- **Thực trạng dữ liệu:** Cột `exp_score` có $3$ giá trị duy nhất (`exp_score min = 0.2019`, `max = 1.0000`). Có $93.64\%$ cặp đạt `exp_score == 1.0` (ứng viên đạt hoặc vượt số năm kinh nghiệm yêu cầu).
- **Thiết lập ngưỡng mới:** $\tau_{\text{pos}} = 1.0000$ (Khớp hoàn toàn) và $\tau_{\text{neg}} = 0.5000$ (Thiếu hụt kinh nghiệm nghiêm trọng >1.7 năm).
- **Kết quả:** 
  - Positive (+1): $4.195$ cặp ($93.64\%$)
  - Negative (-1): $285$ cặp ($6.36\%$)
  - **`Positive ∩ Negative = 0`** (Xác nhận 0% chồng lấp).
- **Ghi chú về ABSTAIN:** Do dữ liệu `exp_score` có $93.64\%$ ở mốc $1.0$ và $6.36\%$ ở mốc $< 0.50$, nên không có mẫu nào nằm ở khoảng giữa $0.50 < \text{exp\_score} < 1.0$.

### Check B: Ngữ nghĩa Đầu ra của LFs (LF Output Semantics)
- Đã xác nhận 100% các LFs xuất ra các giá trị rời rạc hợp lệ trong tập $\{-1, 0, +1\}$.
- **`LF_loc`:** Phản ánh đúng bản chất là hàm nhị phân xác định (Deterministic Binary LF: $1$ nếu cùng thành phố, $-1$ nếu khác thành phố, không có ABSTAIN).
- **`LF_skill`, `LF_sem`, `LF_role`:** Đều hỗ trợ **ABSTAIN Policy (giá trị 0)** với Abstain Rate từ $27\%$ đến $50\%$.

### Check C: Phân tích Phân phối Điều kiện `y_prob` (Monotonicity Verification)
Đã thực hiện nhóm và tính trung bình `y_prob` theo số lượng LFs Dương (+1) mà cặp CV-JD đạt được:

| Số LFs Dương (+1) | Số Lượng Cặp (Count) | `y_prob` Trung Bình (Mean) | `y_prob` Độ Lệch Chuẩn (Std) | `y_prob` [Min, Max] |
| :---: | :---: | :---: | :---: | :---: |
| **0 LFs Positive** | **100** | **0.1125** | 0.0190 | [0.0908, 0.1682] |
| **1 LF Positive** | **1,360** | **0.7326** | 0.1523 | [0.1298, 0.8244] |
| **2 LFs Positive** | **1,923** | **0.8615** | 0.1030 | [0.1822, 0.9909] |
| **3 LFs Positive** | **890** | **0.9308** | 0.0703 | [0.2986, 0.9935] |
| **4 LFs Positive** | **192** | **0.9675** | 0.0396 | [0.9081, 0.9947] |
| **5 LFs Positive** | **15** | **0.9957** | 0.0000 | [0.9957, 0.9957] |

> [!IMPORTANT]
> **Xác nhận Tính Đơn Điệu (Monotonicity Verified):**
> - Khi một cặp CV-JD **không đạt bất kỳ LF Dương nào (0 LFs)**, `y_prob` giảm mạnh xuống **0.1125** (tối thiểu `0.0908`).
> - Khi số LFs Dương tăng từ $1 \rightarrow 5$, `y_prob` tăng đều đặn từ $0.7326 \rightarrow 0.9957$.
> - Điều này chứng minh thuật toán Dawid-Skene EM **không bị Class-Prior Dominated**, mà phản ánh chính xác tín hiệu của các LFs!

### Check D: Đối chiếu Dawid-Skene vs. Consensus Aggregator
- **Hệ số Tương quan Spearman $\rho(y_{\text{DS}}, y_{\text{Consensus}})$:** **$0.9227$** ($p < 10^{-100}$).
- **Sai số Tuyệt đối Trung bình (MAE):** $0.3017$.
- **Đánh giá:** Mô hình Dawid-Skene và mô hình Consensus có độ tương quan rất cao ($> 0.92$), cho thấy tín hiệu tổng hợp rất ổn định.

### Check E: Bảo mật Kiểm thử & Không Rò Rỉ trong `fit()`
- Đã kiểm tra và xác nhận hàm `fit()` của `AspectLabelingFunctions` **CHỈ đọc các trường dữ liệu thô từ `df_train`** (4.480 cặp).
- Tuyệt đối không đọc `df_dev`, `df_test`, hay `legacy_gold_relevance`.

---

## 2. TỔNG KẾT BẢNG TRẠNG THÁI TIẾN ĐỘ

| Pha Nghiên Cứu | Trạng Thái Nghiệm Thu |
| :--- | :--- |
| **Phase 0 — Freeze Baseline** | ✅ **PASS** |
| **Phase 1 — Leakage Removal + Split Manifest** | ✅ **PASS** |
| **Phase 2 — LF + ABSTAIN + Aggregation** | ✅ **PASS** |
| **Phase 2.1 — LF Sanity Check & Distribution Audit** | ✅ **PASS CHÍNH THỨC** |
| **Phase 3 — Human-Validated Benchmark (~100 pairs)** | 🟢 **SẴN SÀNG KHỞI CHẠY** |
| **Phase 4 — Independent Evaluation & Perturbation** | ⏳ Chờ Pha 3 hoàn thành |

Đủ điều kiện nghiệm thu chính thức Pha 2 & Pha 2.1 để chuyển sang **Pha 3: Xây dựng Tập Human-Validated Benchmark (~100 Cặp phân tầng)**!
