# Thực nghiệm

> **Phạm vi bằng chứng.** Hai tệp dữ liệu gốc hiện có và đã được kiểm tra trực tiếp. Tuy nhiên, các artifact thực nghiệm cũ không triển khai đúng toàn bộ giao thức được xác định trong nghiên cứu này: chúng dùng năm tín hiệu (`location`, `skill`, `experience`, `role`, `description`) thay vì ba tín hiệu (`semantic`, `skill`, `experience`), chỉ có mô hình pointwise và pairwise, không có listwise, và dùng toàn bộ tập 100 cặp đã gán nhãn để đánh giá thay vì tách Gold-validation/Gold-test. Vì vậy, các kết quả cũ chỉ được báo cáo như kiểm tra sơ bộ; chúng không được dùng để khẳng định câu hỏi nghiên cứu theo giao thức ba tín hiệu. Các ô chưa có phép đo hợp lệ được ghi rõ là chưa xác định, không được nội suy hoặc thay bằng kết quả của pipeline cũ.

## 1. Dataset và xử lý dữ liệu

### 1.1. Dataset

Nghiên cứu sử dụng **Job Dataset for Recommendation** (Kaggle, tác giả phamtheds), gồm hai bảng độc lập về tin tuyển dụng và hồ sơ ứng viên; dataset không có bảng click, apply, interview hoặc hiring outcome. Audit trực tiếp trên hai tệp CSV cho kết quả ở Bảng 1.

**Bảng 1. Quy mô và schema của dữ liệu gốc.**

| Bảng | Số dòng gốc | Số cột | Khóa định danh | Số thực thể duy nhất | Dòng trùng hoàn toàn | Giá trị thiếu theo CSV |
|---|---:|---:|---|---:|---:|---:|
| Tin tuyển dụng | 14.634 | 19 | `JobID`, `URL Job` | 14.634 job | 0 | 0 |
| Hồ sơ ứng viên | 3.983 | 14 | `UserID`, `URL User` | 3.191 | 792 | 0 |

Bảng tin tuyển dụng có các trường: `JobID`, URL, chức danh, tên và mô tả công ty, quy mô và địa chỉ công ty, mô tả công việc, yêu cầu công việc, phúc lợi, địa điểm làm việc, loại việc làm, yêu cầu giới tính, số lượng tuyển, cấp bậc, số năm kinh nghiệm, lương, hạn nộp và ngành nghề. Các trường được dùng trực tiếp để tạo đặc trưng là `Job Title`, `Job Description`, `Job Requirements`, `Years of Experience` và `Job Address`.

Bảng hồ sơ có các trường: URL, `UserID`, tên ứng viên, ngành nghề, công việc mong muốn, nơi làm việc mong muốn, lương mong muốn, giới tính, tình trạng hôn nhân, tuổi, mục tiêu nghề nghiệp, kỹ năng, bằng cấp và kinh nghiệm làm việc. Các trường được dùng trực tiếp là `Desired Job`, `Target`, `Skills`, `Work Experience` và `Workplace Desired`.

Mặc dù parser CSV không ghi nhận giá trị rỗng, điều này không đồng nghĩa mọi trường đều có cùng chất lượng ngữ nghĩa. Chẳng hạn, kinh nghiệm và lương chứa các giá trị phân loại như “Không yêu cầu kinh nghiệm” hoặc “Thỏa thuận”. Các giá trị này phải được chuẩn hóa bằng quy tắc xác định trước, không được coi là số đo liên tục nguyên trạng.

### 1.2. Làm sạch và chuẩn hóa

Quy trình tiền xử lý được thực hiện trước khi tạo cặp CV–job:

1. **Khử trùng.** Loại 792 dòng CV trùng hoàn toàn theo toàn bộ hàng; đồng thời kiểm tra duy nhất của `UserID` và URL sau khử trùng. Không có job trùng theo toàn bộ hàng, `JobID` hoặc URL. Sau bước này, corpus có 14.634 job và 3.191 CV duy nhất.
2. **Chuẩn hóa văn bản tiếng Việt.** Chuỗi được chuyển về Unicode NFC, chữ thường, chuẩn hóa khoảng trắng và dấu câu; giữ dấu tiếng Việt vì dấu mang thông tin từ vựng. HTML, URL và ký tự điều khiển được loại bỏ. Các biến thể địa danh phổ biến, chẳng hạn “TP.HCM”, “TPHCM” và “Hồ Chí Minh”, được ánh xạ về cùng một dạng chuẩn.
3. **Xử lý trường thiếu về mặt ngữ nghĩa.** Chuỗi rỗng và các placeholder như `nan`, `none`, `null`, `n/a`, “chưa cập nhật” được chuyển thành missing. Một cặp không có đủ văn bản cho một nguồn tín hiệu khiến nguồn đó **abstain**, thay vì tự động bỏ phiếu âm. Không điền văn bản của train bằng nội dung từ validation/test.
4. **Chuẩn hóa kỹ năng.** Trường `Skills` của CV và `Job Requirements` của job được tách theo dấu câu và các marker liệt kê; token chỉ gồm dấu câu bị loại. Từ viết tắt, biến thể viết hoa và biến thể có/không dấu được ánh xạ bằng một từ điển chuẩn hóa kỹ năng được định nghĩa trước, cố định trước thực nghiệm và áp dụng đồng nhất cho mọi partition (không fit trên train). Danh sách cuối cùng được khử trùng trong từng hồ sơ. Chỉ vocabulary TF–IDF và document frequency được fit trên train rồi đóng băng.
5. **Chuẩn hóa kinh nghiệm.** Các khoảng “1–3 năm”, “3–5 năm”, “5–10 năm” được chuyển thành khoảng số; “Không yêu cầu kinh nghiệm” được hiểu là không có ràng buộc tối thiểu và được biểu diễn bằng khoảng \([0,\infty)\) — mọi ứng viên thỏa yêu cầu, không phải giá trị điểm cố định 0 năm — còn “Trên 10 năm” được biểu diễn bằng cận dưới 10. Điểm phù hợp dùng khoảng thay vì giả định mọi ứng viên trong cùng nhóm có đúng một số năm cố định.
6. **Đóng băng bộ biến đổi.** Vocabulary, IDF, từ điển kỹ năng, ngưỡng labeling function và tham số label model chỉ được ước lượng từ train. Validation, Gold-validation và Gold-test chỉ được transform bằng trạng thái đã đóng băng.

### 1.3. Tạo cặp và chia tập

Đơn vị query được chọn là **job**; mỗi query là một tin tuyển dụng và các item là những CV ứng viên. Việc chọn chiều này được giải thích ở Mục 3. Dữ liệu được chia theo `JobID`, không chia ngẫu nhiên theo cặp. Với tỷ lệ 70/15/15, toàn bộ cặp của một job chỉ thuộc đúng một trong ba tập train, validation hoặc test. Ba tập phải thỏa

\[
\mathcal J_{train}\cap\mathcal J_{val}
=\mathcal J_{train}\cap\mathcal J_{test}
=\mathcal J_{val}\cap\mathcal J_{test}=\varnothing.
\]

Để tránh rò rỉ qua biểu diễn CV, giao thức nghiêm ngặt còn yêu cầu CV dùng trong Gold-test không tham gia fit vocabulary, label model hoặc ranker. Tất cả ID của Gold-test được loại khỏi weak-label pool trước khi fit. Mỗi phép bootstrap lấy mẫu lại theo **job**, không lấy mẫu 100 cặp như các quan sát độc lập.

Dataset không có bảng interaction. Vì vậy, không có tín hiệu hành vi để xem là relevance quan sát được; tập Gold do con người gán nhãn là nguồn đánh giá độc lập duy nhất. Artifact hiện có gồm 100 cặp thuộc 12 job và 67 CV, với 4–15 CV/job (trung bình 8,33). Nhãn graded 0–3 có phân phối 30/35/29/6. Tập này hiện do một tác giả gán nhãn theo rubric, không phải recruiter-validated ground truth. Pipeline hỗ trợ file nhãn độc lập của đúng hai annotator và báo exact agreement cùng quadratic-weighted Cohen’s kappa; khi chưa có file thật, audit phải ghi `not_available`.

## 2. Xây dựng tín hiệu và nhãn (Core 1)

Với job \(j\) và CV \(c\), nghiên cứu chỉ dùng ba nguồn tín hiệu phục vụ Core 1.

### 2.1. Tín hiệu semantic

Văn bản phía job được tạo bởi

\[
t_j = \texttt{Job Title}\;\Vert\;\texttt{Job Description}\;\Vert\;\texttt{Job Requirements},
\]

và văn bản phía CV bởi

\[
t_c = \texttt{Desired Job}\;\Vert\;\texttt{Target}\;\Vert\;\texttt{Skills}.
\]

Hai văn bản được mã hóa bằng model sentence embedding đa ngôn ngữ pretrained đã khóa trong cấu hình (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`). Vector được chuẩn hóa L2 và độ tương đồng semantic là tích vô hướng, tương đương cosine similarity:

\[
s_{sem}(j,c)=\mathbf e_j^\top\mathbf e_c,
\qquad \lVert\mathbf e_j\rVert_2=\lVert\mathbf e_c\rVert_2=1.
\]

Model embedding không được fine-tune bằng Gold. Mỗi job/CV chỉ được encode một lần theo batch rồi ánh xạ về các pair. TF–IDF vẫn được fit trên development-train nhưng chỉ dùng cho role/description **lexical components** của baseline thủ công `Score_H`, không được gọi là tín hiệu semantic.

### 2.2. Tín hiệu kỹ năng

Gọi \(K_j\) là tập kỹ năng chuẩn hóa trích từ yêu cầu công việc và \(K_c\) là tập kỹ năng của CV. Điểm skill dùng Jaccard/IoU:

\[
s_{skill}(j,c)=\frac{|K_j\cap K_c|}{|K_j\cup K_c|}.
\]

Nếu cả hai tập rỗng, nguồn skill abstain thay vì gán 0; nếu chỉ một phía rỗng sau khi đã xác nhận trường dữ liệu có mặt, điểm bằng 0.

### 2.3. Tín hiệu kinh nghiệm

Gọi khoảng kinh nghiệm yêu cầu là \([l_j,u_j]\) và khoảng kinh nghiệm ứng viên là \([l_c,u_c]\). Khoảng cách giữa hai khoảng là

\[
d_{exp}(j,c)=\max(0,l_j-u_c,l_c-u_j),
\]

và điểm phù hợp được chuẩn hóa bởi

\[
s_{exp}(j,c)=\exp\!\left(-\frac{d_{exp}(j,c)}{\max(1,l_j)}\right).
\]

Cách định nghĩa này cho điểm 1 khi hai khoảng giao nhau và giảm trơn khi khoảng cách tăng.

### 2.4. Labeling functions và label model

Mỗi nguồn \(m\in\{sem,skill,exp\}\) được chuyển thành labeling function \(\lambda_m\in\{-1,0,+1\}\). Ngưỡng âm \(\tau_m^-\) là phân vị 25 trên các giá trị active của development-train. Ngưỡng dương \(\tau_m^+\) là phân vị 75 trên positive tail \(\{s_m:s_m>\tau_m^-\}\), không phải trên toàn bộ phân phối. Chính sách tie-aware này tránh trường hợp skill overlap thưa có phân vị 25 và 75 cùng bằng 0 rồi fallback về giá trị cực đại, khiến LF gần như không bao giờ bỏ phiếu dương. Cả hai ngưỡng chỉ dùng development-train và được đóng băng trước mọi held-out inference:

\[
\lambda_m(j,c)=
\begin{cases}
+1,&s_m\ge \tau_m^+,\\
-1,&s_m\le \tau_m^-,\\
0,&\tau_m^-<s_m<\tau_m^+\text{ hoặc nguồn thiếu/không chắc chắn.}
\end{cases}
\]

Ba phiếu được tổng hợp bằng label model latent-variable kiểu Dawid–Skene. Với relevance nhị phân ẩn \(Y\), mô hình ước lượng prior \(\pi=P(Y=1)\), sensitivity \(\alpha_m=P(\lambda_m=1\mid Y=1)\) và specificity \(\beta_m=P(\lambda_m=-1\mid Y=0)\) bằng EM trên train. Xác suất yếu của cặp là

\[
\tilde p_{jc}=P(Y=1\mid\lambda_{sem},\lambda_{skill},\lambda_{exp}).
\]

Các phiếu abstain không tham gia likelihood của nguồn tương ứng. \(\tilde p_{jc}\) luôn được gọi là **weak/estimated probability**, không phải nhãn thật.

Dawid–Skene giả định các nguồn độc lập có điều kiện theo \(Y\). Trước khi dùng posterior, nghiên cứu báo cáo coverage, conflict và tương quan Spearman biên (marginal/pairwise-dependence diagnostic) giữa từng cặp labeling functions trên tập hợp các cặp cùng bỏ phiếu (joint non-abstention); đây không chứng minh độc lập có điều kiện. Tiêu chí định trước là không có cặp nguồn nào có \(|\rho|\ge0{,}7\); nếu vượt ngưỡng, kết quả label model phải được xem là không đáng tin nếu chưa mô hình hóa phụ thuộc. Audit lịch sử với năm nguồn cho tương quan lớn nhất 0,138, nhưng con số này **không thay thế** audit phải chạy lại với đúng ba nguồn và dữ liệu đã xử lý theo giao thức hiện tại.

Baseline tạo nhãn là luật **3/3 nguồn đồng ý**: nhãn dương nếu cả ba LF bỏ phiếu dương, nhãn âm nếu cả ba bỏ phiếu âm, và abstain trong các trường hợp còn lại. Luật này có precision kỳ vọng cao nhưng coverage thấp; nó chỉ là đối chứng, không dùng làm nhãn cuối.

**Giả thuyết H-label.** Label model có precision và recall trên Gold-validation cao hơn luật 3/3 ở cùng ngưỡng quyết định. Baseline là luật 3/3; metric là precision và recall theo cặp trên Gold-validation; giả thuyết được ủng hộ khi label model tăng recall mà precision không giảm quá 0,02. Tiêu chí này phải được khóa trước khi xem Gold-validation.

## 3. Thiết lập thực nghiệm

### 3.1. Đơn vị ranking

Nghiên cứu dùng **job làm query** và CV làm item. Lý do nghiệp vụ là bài toán xếp danh sách ứng viên cho một tin tuyển dụng; lý do thống kê là corpus có 14.634 job và 3.191 CV duy nhất, nên với một candidate pool cố định, mỗi job có thể được đánh giá trên cùng số CV. Artifact Gold hiện có cũng được tổ chức theo job, với trung bình 8,33 CV/query. Dataset không chứa interaction để xây dựng danh sách job thực sự đã được một CV xem hoặc ứng tuyển, nên chiều CV→job không có candidate list quan sát được và sẽ phụ thuộc hoàn toàn vào chiến lược sinh âm tính.

### 3.2. Baseline và phương pháp

Baseline bắt buộc là công thức trọng số thủ công đang được kiểm tra:

\[
Score_H=0{,}30S_{location}+0{,}25S_{skill}+0{,}20S_{experience}
+0{,}15S_{role}+0{,}10S_{description}.
\]

Phương pháp đề xuất nhận đúng vector ba tín hiệu

\[
\mathbf x_{jc}=[s_{sem},s_{skill},s_{exp}]
\]

và được huấn luyện bằng weak probability \(\tilde p_{jc}\). Ba formulation được so sánh trên validation:

- **Pointwise:** mô hình tuyến tính với logit \(f_\theta(\mathbf x)\), tối ưu soft binary cross-entropy
  \[
  \mathcal L_{point}=-\sum_i\big[\tilde p_i\log\sigma(f_i)+(1-\tilde p_i)\log(1-\sigma(f_i))\big].
  \]
- **Pairwise:** RankNet tạo cặp trong cùng job khi \(|\tilde p_i-\tilde p_j|\ge0{,}02\), tối ưu
  \[
  \mathcal L_{pair}=\sum_{(i,j):\tilde p_i>\tilde p_j}
  \log(1+\exp(-(f_i-f_j))).
  \]
- **Listwise:** ListNet tối ưu cross-entropy giữa phân phối weak relevance và phân phối score trong từng query:
  \[
  P_i=\frac{\exp(\tilde p_i/T)}{\sum_{r\in q}\exp(\tilde p_r/T)},\quad
  Q_i=\frac{\exp(f_i)}{\sum_{r\in q}\exp(f_r)},\quad
  \mathcal L_{list}=-\sum_q\sum_{i\in q}P_i\log Q_i.
  \]

Cùng kiến trúc scorer và cùng ba feature được dùng giữa các formulation để khác biệt phản ánh objective thay vì năng lực mô hình.

### 3.3. Metric và tiêu chí kết luận

Ranking được đánh giá bằng nDCG@5, nDCG@10 và MRR. nDCG dùng đầy đủ grade 0–3; MRR xem grade \(\ge2\) là relevant. Mỗi metric được tính riêng theo job rồi macro-average qua job. Chất lượng weak label được đánh giá riêng bằng precision và recall trên Gold-validation với relevance nhị phân \(y=1[grade\ge2]\). F1 của label model, nếu báo cáo bổ sung, không được dùng làm bằng chứng cho chất lượng ranking.

**Giả thuyết chính H-main.** Ranker được chọn trên validation có nDCG@5, nDCG@10 và MRR cao hơn \(Score_H\) trên Gold-test. Baseline là \(Score_H\); đơn vị thống kê là job; khoảng tin cậy 95% được tính bằng paired bootstrap theo job sau khi trung bình qua năm seed 11, 23, 42, 67 và 89. Bằng chứng ủng hộ yêu cầu chênh lệch trung bình dương và CI 95% không chứa 0 cho metric chính nDCG@5. Các metric còn lại xác nhận tính nhất quán, không thay tiêu chí chính sau khi xem kết quả.

**Giả thuyết H-form.** Pointwise, pairwise và listwise không được giả định trước phương pháp thắng. Formulation có validation nDCG@5 cao nhất được chọn; hòa trong sai số 0,005 được giải bằng validation pairwise loss thấp hơn, rồi bằng mô hình đơn giản hơn. Gold-test không tham gia lựa chọn này.

### 3.4. Gold-validation và Gold-test

Tập Gold phải được tách một lần theo job, không theo cặp. Policy hiện tại chỉ sort/shuffle job ID bằng seed cố định và không đọc relevance: 4 job vào Gold-validation theo cấu hình, mọi job còn lại vào Gold-test. Gold-validation chỉ dùng để đánh giá label-model prerequisite tại ngưỡng posterior cố định 0,5 và chọn formulation LTR; hyperparameter nội bộ được chọn trên weak-validation. Gold-test được mở đúng một lần sau khi khóa toàn bộ pipeline.

Artifact hiện có đã chấm cả 100 cặp như một tập đánh giá duy nhất, vì vậy **không đáp ứng** giao thức Gold-validation/Gold-test này. Các kết quả ở Mục 4–6 được phân biệt rõ giữa “kết quả hợp lệ theo giao thức” và “kiểm tra sơ bộ lịch sử”.

## 4. Kết quả chính

**Bảng 2. Kết quả cần dùng để trả lời câu hỏi nghiên cứu chính trên Gold-test.**

| Phương pháp | nDCG@5 | nDCG@10 | MRR | Chênh nDCG@5 so với \(Score_H\), CI 95% |
|---|---:|---:|---:|---:|
| \(Score_H\) thủ công | Chưa có phép đo Gold-test hợp lệ | Chưa có phép đo Gold-test hợp lệ | Chưa có phép đo Gold-test hợp lệ | — |
| LTR ba tín hiệu, formulation được chọn | Chưa chạy | Chưa chạy | Chưa chạy | Chưa xác định |

Do chưa có run dùng đúng ba nguồn, đủ ba formulation và Gold-test giữ kín, câu hỏi nghiên cứu chính **chưa thể kết luận**. Việc điền số của pipeline năm tín hiệu vào Bảng 2 sẽ làm thay đổi phương pháp được kiểm định và vi phạm giao thức đã xác định.

Để minh bạch, Bảng 3 ghi lại kiểm tra sơ bộ lịch sử trên toàn bộ 100 cặp Gold. Đây không phải kết quả xác nhận của nghiên cứu hiện tại.

**Bảng 3. Kết quả sơ bộ của pipeline cũ dùng năm tín hiệu và toàn bộ Gold (12 job, 5 seed).**

| Mô hình lịch sử | nDCG@5 | nDCG@10 | MRR† |
|---|---:|---:|---:|
| \(Score_H\) | 0,8342 | 0,8690 | 0,4778 |
| Pointwise soft BCE (B2) | 0,8543 | 0,9008 | 0,6278 |
| Pairwise RankNet (M1) | 0,8395 | 0,8759 | 0,5267 ± 0,0472 |

† MRR được tính lại từ prediction artifact với grade \(\ge2\) là relevant. Đối với M1 so với \(Score_H\), chênh lệch nDCG@5 là +0,0053 với paired-bootstrap CI 95% [−0,0155; 0,0270]; chênh lệch nDCG@10 là +0,0069 [−0,0111; 0,0273]; chênh lệch MRR là +0,0489 [0,0000; 0,1133]. Tất cả CI đều chứa hoặc chạm 0. Do đó, ngay cả trong pipeline cũ, bằng chứng không đủ để khẳng định RankNet xếp hạng tốt hơn công thức thủ công.

## 5. Kết quả điều kiện

### 5.1. Câu hỏi phụ 1: chất lượng label model

**Giả thuyết.** Dawid–Skene ba nguồn tăng coverage/recall so với luật 3/3 mà không làm precision giảm quá 0,02. Baseline là 3/3; metric là precision và recall trên Gold-validation; ngưỡng posterior được khóa cố định ở 0,5 và không được chọn trên Gold-validation.

**Bảng 4. Chất lượng weak label trên Gold-validation.**

| Bộ tạo nhãn | Precision | Recall | Coverage |
|---|---:|---:|---:|
| Luật 3/3 | Chưa chạy | Chưa chạy | Chưa chạy |
| Label model ba nguồn | Chưa chạy | Chưa chạy | Chưa chạy |

Các artifact cũ không báo cáo phép so sánh này với đúng ba nguồn và không có Gold-validation tách riêng. Vì vậy, chưa có bằng chứng rằng weak probability đủ chính xác để dùng làm target; đây là điều kiện cần được thỏa trước khi diễn giải kết quả ranking.

### 5.2. Câu hỏi phụ 2: pointwise, pairwise hay listwise

**Giả thuyết.** Objective khai thác cấu trúc trong query sẽ cải thiện nDCG@5 trên validation so với pointwise. Baseline là pointwise; metric chọn mô hình là validation nDCG@5, với pair-loss làm tie-break như Mục 3.

**Bảng 5. So sánh formulation trên validation.**

| Formulation | Validation nDCG@5 | Validation nDCG@10 | Validation MRR | Quyết định |
|---|---:|---:|---:|---|
| Pointwise | Chưa chạy theo giao thức ba tín hiệu | Chưa chạy | Chưa chạy | Chưa xác định |
| Pairwise | Chưa chạy theo giao thức ba tín hiệu | Chưa chạy | Chưa chạy | Chưa xác định |
| Listwise | Chưa được triển khai | Chưa được triển khai | Chưa được triển khai | Chưa xác định |

Do thiếu listwise và thiếu phép so sánh đồng nhất trên ba tín hiệu, câu hỏi phụ 2 chưa thể trả lời. Không thể dùng chênh lệch giữa B2 và M1 của pipeline cũ làm kết luận vì hai mô hình lịch sử không được chọn theo giao thức Gold-validation đã nêu.

## 6. Ablation

Chỉ thực hiện đúng hai ablation đã xác định; không thêm phân tích theo domain, hard-negative hoặc sensitivity tham số.

### 6.1. Bỏ label model

**Giả thuyết.** Việc học độ tin cậy khác nhau của ba nguồn bằng label model tạo target tốt hơn trung bình cộng đơn giản. Ablation thay

\[
\tilde p_{jc}=P(Y=1\mid\lambda_{sem},\lambda_{skill},\lambda_{exp})
\]

bằng trung bình cộng có trọng số đều của ba tín hiệu **sau khi đã điền thiếu (median-imputed) bằng thống kê của train**:

\[
\bar s_{jc}=\frac{s_{sem}^{imp}+s_{skill}^{imp}+s_{exp}^{imp}}{3},
\]

sau đó giữ nguyên formulation LTR đã chọn, split, kiến trúc và seed. Metric là nDCG@5 trên Gold-test; Core 1 được ủng hộ khi mô hình đầy đủ có chênh lệch dương với CI 95% không chứa 0. `manual_score_h` được giữ nguyên là baseline lịch sử 5 thành phần (location/skill/experience/role/description) và không phải so sánh có kiểm soát trên cùng ba tín hiệu; baseline có kiểm soát `manual_3signal` với trọng số cố định trên \([s_{sem},s_{skill},s_{exp}]\) được ghi nhận là mở rộng ngoài phạm vi freeze hiện tại.

### 6.2. Bỏ LTR

**Giả thuyết.** Học một hàm xếp hạng theo query cải thiện thứ tự so với xếp trực tiếp bằng weak probability. Ablation sắp CV theo \(\tilde p_{jc}\), không huấn luyện ranker. Label model, dữ liệu và Gold-test được giữ nguyên. Metric và tiêu chí kết luận giống ablation thứ nhất; Core 2 được ủng hộ khi LTR vượt direct probability ranking với CI 95% không chứa 0.

**Bảng 6. Hai ablation bắt buộc trên Gold-test.**

| Hệ thống | Label target | Cơ chế ranking | nDCG@5 | nDCG@10 | MRR |
|---|---|---|---:|---:|---:|
| Đầy đủ | Label model | LTR được chọn | Chưa chạy | Chưa chạy | Chưa chạy |
| Bỏ Core 1 | Trung bình ba nguồn | Cùng LTR | Chưa chạy | Chưa chạy | Chưa chạy |
| Bỏ Core 2 | Label model | Xếp trực tiếp theo \(\tilde p\) | Chưa chạy | Chưa chạy | Chưa chạy |

Không có artifact lịch sử nào tương ứng chính xác với hai ablation này; do đó không báo cáo số thay thế.

## 7. Hạn chế

Thứ nhất, Gold hiện chỉ có 100 cặp trên 12 job, tương đương trung bình 8,33 item/query. Sau khi tách Gold-validation/Gold-test, số query kiểm định còn nhỏ hơn nữa; paired-bootstrap vì vậy có power thấp và CI dự kiến rộng. Các cặp trong cùng job không được xem là quan sát độc lập.

Thứ hai, benchmark hiện do một tác giả gán nhãn và chưa có file annotator thứ hai, nên audit IAA phải là `not_available`. Nó là **author-annotated benchmark**, không phải ground truth tuyển dụng. Pipeline đã khóa schema để tính exact agreement và quadratic-weighted Cohen’s kappa khi có nhãn độc lập thật, nhưng code không thay thế được việc thu thập annotation. Phân phối grade 3 chỉ có 6/100 cặp, làm ước lượng chất lượng ở mức relevance cao kém ổn định.

Thứ ba, giả thiết độc lập có điều kiện giữa semantic, skill và experience có thể không đúng hoàn toàn: mô tả semantic thường chứa kỹ năng và kinh nghiệm. Tương quan LF thấp không chứng minh độc lập có điều kiện; nó chỉ là kiểm tra chẩn đoán tối thiểu. Weak probability còn được tạo từ các feature mà ranker nhận làm đầu vào, nên mô hình có thể học lại cấu trúc labeling functions. Đánh giá Gold độc lập là cơ chế chính để hạn chế vòng lặp này, không phải bằng chứng rằng weak label là nhãn thật.

Thứ tư, dữ liệu không có click/apply/hiring outcome. Nghiên cứu chỉ đánh giá mức phù hợp nội dung do người gán nhãn nhận định, không chứng minh khả năng dự đoán ứng tuyển, phỏng vấn, tuyển dụng hoặc hiệu quả kinh doanh.

Thứ năm, corpus chủ yếu là tin tuyển dụng và CV tiếng Việt thu thập từ các nền tảng việc làm Việt Nam. Kết luận, nếu có, chỉ áp dụng cho ngôn ngữ, schema và phân phối ngành nghề của corpus này; chưa có bằng chứng về khả năng chuyển sang ngôn ngữ, quốc gia hoặc hệ thống taxonomy nghề nghiệp khác.

Cuối cùng, kết quả lịch sử dùng subsample 160 job, 240 CV và tối đa 100 CV/job, không phải toàn bộ tích Descartes của 14.634 job và 3.191 CV. Chúng còn dùng năm tín hiệu và toàn bộ Gold để đánh giá. Vì vậy, các số sơ bộ chỉ cho thấy pipeline có thể chạy và chưa cho phép khẳng định hai core của thiết kế ba tín hiệu.