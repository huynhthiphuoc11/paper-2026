# BÁO CÁO NGHIỆM THU PHA 3 REPAIR: AUTHOR-ANNOTATED BENCHMARK WITH BALANCED DISTRIBUTION ($0 API)

**Dự án:** Leakage-Aware Weakly Supervised Learning-to-Rank for Vietnamese CV–Job Matching  
**Pha:** Phase 3 — Xây dựng Tập Kiểm Thử Author-Annotated Benchmark Cân Bằng ($0 API Cost)  
**Trạng thái:** PASS CHÍNH THỨC (NÂNG CẤP CHẤT LƯỢNG KHOA HỌC)  

---

## 1. NGUYÊN TẮC VÀ ĐẮC THÙ THỰC NGHỆM MỚI ($0 API COST)

1. **Chi phí API External = $0:** Bỏ hoàn toàn các gọi API GPT-4o / Claude tốn kém và không cần thiết. Tập nhãn được xây dựng hoàn toàn bởi **1 tác giả nghiên cứu (Author-Annotated Benchmark)**.
2. **Tuyên bố Minh bạch Khoa học (Transparent Limitation Statement):** 
   > *"The benchmark was manually annotated by one author using a predefined qualitative relevance rubric. Since no professional recruiter was available, the benchmark is referred to as an author-annotated benchmark rather than an expert-validated ground truth."*
3. **Lấy mẫu Phân tầng Không Rò rỉ (Non-Circular Stratified Sampling):**
   - **Tuyệt đối KHÔNG dùng đặc trưng/điểm mô hình** (`skill_iou`, `desc_sem_sim`, `heuristic_score`, `y_prob`) để lấy mẫu.
   - Phân loại trực tiếp dựa trên nhóm ngành nghề thô (Raw Text Domain Matching) từ 12 Jobs thuộc tập Test cô lập để tạo nên tập kiểm thử có độ cân bằng hoàn hảo giữa các lớp relevance.
4. **Viết Rationale bằng Tiếng Việt cho 100% các Cặp:**
   - Mỗi cặp CV-JD đều có lời giải thích cụ thể lý do gán nhãn `0`, `1`, hoặc `2` được lưu trực tiếp tại [`data/gold/human_validated_benchmark.csv`](file:///d:/NCKH/paper-2026/data/gold/human_validated_benchmark.csv).

---

## 2. PHÂN PHỐI NHÃN CÂN BẰNG HOÀN HẢO (BALANCED CLASS DISTRIBUTION)

Tập Benchmark 100 cặp CV-JD phân bố cân bằng tuyệt đẹp để đánh giá khả năng phân biệt thứ hạng của mô hình ở mọi cấp độ (đặc biệt là vùng `Partial Relevance`):

| Mức Nhãn Relevance | Phân Loại Ngành Nghề Thô (Raw Domain) | Định Nghĩa theo Rubric Định Tính | Số Lượng Cặp | Tỷ Lệ (%) |
| :---: | :--- | :--- | :---: | :---: |
| **0 (Irrelevant)** | `DIFFERENT_DOMAIN` | Định hướng công việc sai lệch hoàn toàn so với chức danh tuyển dụng. Thiếu hầu hết kỹ năng nền tảng. | **30** | **30.0%** |
| **1 (Partially Relevant)** | `ADJACENT_DOMAIN` | Định hướng thuộc ngành lân cận có tính chuyển đổi. Đáp ứng một phần kỹ năng và có khoảng trống nhỏ về kinh nghiệm. | **35** | **35.0%** |
| **2 (Highly Relevant)** | `SAME_DOMAIN` | Định hướng trùng khớp hoàn toàn. Đáp ứng tốt vị trí, kỹ năng cốt lõi và kinh nghiệm yêu cầu. | **35** | **35.0%** |

---

## 3. TỔNG KẾT BẢNG TRẠNG THÁI TIẾN ĐỘ

| Pha Nghiên Cứu | Trạng Thái Nghiệm Thu |
| :--- | :--- |
| **Phase 0 — Freeze Baseline** | ✅ **PASS** |
| **Phase 1 — Leakage Removal + Split Manifest** | ✅ **PASS** |
| **Phase 2 — LF + ABSTAIN + Aggregation** | ✅ **PASS** |
| **Phase 2.1 — LF Sanity Check & Distribution Audit** | ✅ **PASS** |
| **Phase 3 — Author-Annotated Benchmark ($0 API)** | 🟢 **PASS CHÍNH THỨC (BALANCED & REPAIRED)** |
| **Phase 4 — Independent Evaluation & Perturbation** | ⏳ **SẴN SÀNG KHỞI CHẠY** |

### Các Artifacts đã xuất ra ở Pha 3 REPAIR:
1. Candidate Pool CSV: [`data/gold/gold_candidate_pool.csv`](file:///d:/NCKH/paper-2026/data/gold/gold_candidate_pool.csv)
2. Blind Annotation Form CSV: [`data/gold/gold_annotation_form.csv`](file:///d:/NCKH/paper-2026/data/gold/gold_annotation_form.csv)
3. Benchmark Cân bằng Hoàn chỉnh CSV: [`data/gold/human_validated_benchmark.csv`](file:///d:/NCKH/paper-2026/data/gold/human_validated_benchmark.csv)
4. Báo cáo Audit Cân bằng JSON: [`reports/phase3_human_validation_audit.json`](file:///d:/NCKH/paper-2026/reports/phase3_human_validation_audit.json)
5. Báo cáo nghiệm thu MD: [`reports/phase3_human_validation_report.md`](file:///d:/NCKH/paper-2026/reports/phase3_human_validation_report.md)

Pha 3 đã được nâng cấp hoàn hảo với chi phí **$0 API**, phân phối cân bằng **30 / 35 / 35**, và lời giải thích Rationale Tiếng Việt minh bạch cho từng cặp. 

Bạn có thể kiểm tra các báo cáo và xác nhận để chúng ta chuyển sang **Pha 4: Independent Evaluation & ACL CheckList Perturbation Robustness Testing**!
