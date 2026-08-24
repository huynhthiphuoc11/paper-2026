# QUY TRÌNH GÁN NHÃN VÀ THỂ THỨC ĐÁNH GIÁ HUMAN-VALIDATED BENCHMARK (PHASE 3 ANNOTATION PROTOCOL)

**Dự án:** Leakage-Aware Weakly Supervised Learning-to-Rank for Vietnamese CV–Job Matching  
**Pha:** Phase 3 — Human-Validated Benchmark Construction by Explicit Rubric  

---

## 1. NGUYÊN TẮC CỐT LÕI (CORE PRINCIPLES)

1. **Thành thật Khoa học (Academic Transparency):** Tập nhãn được gọi tên chính xác là **Human-Validated Benchmark dựa trên Rubric công khai** do tác giả thực hiện (Author-Annotated Benchmark), không tuyên bố sai sự thật là "Ground Truth từ Chuyên gia Tuyển dụng".
2. **Quy trình Blind Tuyệt đối (Strict Blind Annotation):** Người gán nhãn CHỈ xem nội dung văn bản thô (Raw Job Description, Requirements, Title, Address vs. Raw Candidate Desired Job, Skills, Experience, Workplace). **Tuyệt đối không truy cập:** điểm số đặc trưng (`skill_iou`, `desc_sem_sim`), điểm Heuristic, nhãn yếu `y_prob`, hay dự đoán của các mô hình.
3. **Chất lượng hơn Số lượng (Quality over Quantity):** Tập 100 cặp phân tầng (30 Easy Positives, 30 Easy Negatives, 40 Boundary Cases) được đánh giá kỹ lưỡng kèm Rationale giải thích ngắn 1-2 câu.
4. **Không điều chỉnh nhãn ép con số $\kappa$:** Con số Cohen's $\kappa$ phản ánh thực tế sự đồng thuận giữa Con người và LLMs (GPT-4o, Claude 3.5), không chỉnh sửa nhãn để đạt ngưỡng hình thức.

---

## 2. BẢNG RUBRIC ĐÁNH GIÁ THỨ BẬC ĐỊNH TÍNH (QUALITATIVE ORDINAL RUBRIC)

Annotator đánh giá từng cặp CV-JD trên 4 khía cạnh trước khi đưa ra nhãn tổng thể (Overall Relevance):

| Khía Cạnh | Mức 0 (Không phù hợp / Lệch hẳn) | Mức 1 (Phù hợp một phần / Chuyển đổi) | Mức 2 (Rất phù hợp / Trùng khớp) |
| :--- | :--- | :--- | :--- |
| **Role & Occupation** | Trái ngược định hướng/chức danh hoàn toàn | Ngành nghề lân cận, có tính chuyển đổi tốt | Trùng khớp chức danh và ngành nghề mong muốn |
| **Core Skills** | Thiếu phần lớn kỹ năng cốt lõi bắt buộc | Đáp ứng một phần kỹ năng yêu cầu | Đáp ứng phần lớn/toàn bộ kỹ năng cốt lõi |
| **Work Experience** | Không đạt yêu cầu kinh nghiệm đáng kể | Gần đạt (thiếu nhẹ < 1.5 năm) | Đạt hoặc vượt yêu cầu số năm kinh nghiệm |
| **Workplace Location** | Địa điểm không phù hợp, khó di chuyển | Địa điểm lân cận hoặc có chế độ Remote | Trùng khớp địa điểm làm việc yêu cầu |

### Nhãn Tổng thể (Overall Relevance Score):
* **`0` (Irrelevant):** Ứng viên sai lệch định hướng nghề nghiệp HOẶC thiếu hầu hết kỹ năng nền tảng.
* **`1` (Partially Relevant):** Ứng viên đáp ứng một phần yêu cầu, có kỹ năng chuyển đổi tốt hoặc thiếu nhẹ kinh nghiệm.
* **`2` (Highly Relevant):** Ứng viên đáp ứng xuất sắc vị trí, kỹ năng cốt lõi và kinh nghiệm yêu cầu.

---

## 3. CHƯƠNG TRÌNH ANNOTATION 2 VÒNG (2-ROUND ANNOTATION PROCESS)

### Vòng 1: Thử nghiệm Pilot (20 cặp đầu tiên)
- Người gán nhãn thực hiện đánh giá 20 cặp đầu tiên dựa trên Rubric để phát hiện các tình huống ranh giới mơ hồ (Boundary Ambiguities).
- Đóng băng (lock) quy tắc Rubric sau Vòng 1.

### Vòng 2: Gán nhãn Toàn bộ (80 cặp còn lại)
- Thực hiện gán nhãn nốt 80 cặp còn lại theo đúng Rubric đã đóng băng.
- Ghi chú Rationale giải thích lý do gán nhãn cho từng cặp.

---

## 4. QUY TRÌNH AUDIT ĐỘC LẬP BẰNG LLMs (INDEPENDENT LLM CROSS-VALIDATION)

- Truyền **nội dung văn bản thô (Raw CV + Raw JD)** và bảng Rubric định tính cho **GPT-4o** và **Claude 3.5 Sonnet** chạy độc lập (Temperature = 0).
- Báo cáo 3 chỉ số **Cohen's $\kappa$**:
  1. $\kappa(\text{Human}, \text{GPT-4o})$
  2. $\kappa(\text{Human}, \text{Claude})$
  3. $\kappa(\text{GPT-4o}, \text{Claude})$
- Các trường hợp bất đồng (Disagreement) sẽ được lưu vết để phân tích định chất (Qualitative Error Analysis).
