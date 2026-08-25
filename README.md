# Three-Signal CV–Job Matching Experiments

Pipeline thực nghiệm cho nghiên cứu weakly supervised Learning-to-Rank trên dữ liệu CV–job tiếng Việt. Giao thức khoa học được mô tả tại [EXPERIMENTS.md](EXPERIMENTS.md).

## Entry point chính

Notebook chạy full authoritative:

- [notebooks/experiments_3signal.ipynb](notebooks/experiments_3signal.ipynb)

Notebook là tài liệu thực nghiệm độc lập: toàn bộ code audit, feature engineering, Dawid–Skene, pointwise/pairwise/listwise LTR, protocol lock, Gold-test, ablation và paired bootstrap nằm trực tiếp trong các cell. Notebook không gọi logic thực nghiệm từ [src/experiments/](src/experiments/); thư mục đó chỉ còn là bản module hóa phục vụ kiểm thử và đối chiếu. Mỗi class được định nghĩa nguyên khối với method nằm trực tiếp trong class body; notebook không dùng `class: pass`, generated wrapper method hoặc monkey-patching.

Full run lấy mẫu 2.000 job và 1.500 CV, ghép 200 CV cho mỗi job (400.000 pair trước split). Các pair không được coi là 400.000 quan sát thống kê độc lập; split, metric và bootstrap đều sử dụng job/query làm đơn vị.

Để kiểm soát overfitting, scorer được giữ tuyến tính với đúng ba feature, dùng L2 weight decay `1e-4`, gradient clipping và query-disjoint development-validation. Early stopping và hyperparameter selection tối đa hóa macro weak-validation nDCG@5; objective loss và generalization gap chỉ là diagnostic. Development-test được mở đúng một lần sau selection như final weak-ranking internal check và không được dùng để thay đổi formulation/checkpoint. Training dùng CUDA bắt buộc theo cấu hình; model, feature/target tensor, listwise query index và RankNet pair tensors nằm trên cùng GPU. Dữ liệu được materialize một lần cho mỗi split, còn RankNet pair state được tái sử dụng trong toàn bộ hyperparameter grid của cùng seed.

Notebook trình bày tuần tự các ranh giới của giao thức:

1. Audit dữ liệu và Gold identity.
2. Gold split label-blind với 4 job validation theo cấu hình; mọi job còn lại thuộc Gold-test.
3. Train-only lexical baselines và đúng ba tín hiệu: multilingual sentence-embedding semantic, skill overlap và experience compatibility.
4. Ba labeling functions và Dawid–Skene.
5. Pointwise, pairwise và listwise LTR.
6. Fixed posterior threshold 0,5; formulation-specific batch units; protocol lock trước khi mở Gold-test đúng một lần.
7. Kết quả chính và đúng hai ablation bắt buộc.

## Cài đặt

Pipeline chính thức yêu cầu NVIDIA GPU, driver tương thích và PyTorch CUDA build. File `requirements.txt` chỉ khai báo version tối thiểu; trên Windows cần cài wheel CUDA từ index chính thức của PyTorch trước, rồi cài phần còn lại:

```bash
python -m pip uninstall -y torch
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Xác nhận môi trường trước khi mở notebook:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Cấu hình CUDA không fallback sang CPU. Nếu GPU/driver/PyTorch CUDA không khả dụng, runner dừng trước khi tạo run directory hoặc xử lý Gold.

## Chạy full authoritative

Khởi động kernel sạch và chạy **Run All** trên notebook đúng một lần. Notebook luôn dùng cấu hình full trong `configs/experiment_3signal.yaml`; không có công tắc smoke.

Có thể thực thi từ terminal bằng đúng Python đã cài dependencies:

```bash
python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

source = Path("notebooks/experiments_3signal.ipynb")
target = Path("notebooks/experiments_3signal_full_executed.ipynb")
notebook = nbformat.read(source, as_version=4)
NotebookClient(
    notebook,
    timeout=7200,
    kernel_name="python3",
    resources={"metadata": {"path": str(Path.cwd())}},
).execute()
nbformat.write(notebook, target)
PY
```

Mỗi full run tạo một thư mục bất biến `results/three_signal/full/<protocol_sha256>/`; nếu hash đã có, run mới dùng hậu tố `-run-002`, `-run-003`,… và không ghi đè artifact cũ. Toàn bộ checkpoint chính và ablation được tạo, băm và ghi vào `audit/protocol_lock.json` trước khi Gold-test được mở. Gate Gold-test dùng sentinel bền vững `audit/gold_test_opened.json`. Nếu điều kiện chất lượng label model trên Gold-validation không đạt, notebook chủ động dừng trước formulation training và không mở Gold-test. Mỗi run ghi `audit/environment_manifest.json` gồm Python, platform, device policy và version các package cốt lõi; protocol lock chứa nội dung cùng SHA-256 của manifest này.

`s_sem` dùng model pretrained đa ngôn ngữ được khóa trong cấu hình; lần chạy đầu có thể tải model từ Hugging Face và các lần sau dùng cache cục bộ. TF-IDF chỉ còn phục vụ role/description lexical components của baseline `Score_H`, không được mô tả là semantic embedding.

Gold split là label-blind và query-disjoint nhưng không còn khóa cứng tổng cộng 12 job. `gold.validation_jobs` quy định số job validation, phần còn lại là Gold-test; mọi thay đổi cấu hình hoặc Gold làm đổi protocol hash. `gold.independent_annotations_path` là tùy chọn: khi chưa có nhãn độc lập thật, artifact `audit/inter_annotator_agreement.json` báo `not_available`; khi có đúng hai annotator, pipeline báo exact agreement và quadratic-weighted Cohen’s kappa. Nhãn độc lập không tham gia tuning hoặc chọn formulation.

Gold split hiện dùng policy label-blind: chỉ sort/shuffle `job_id` bằng seed cố định, không đọc `relevance`. File split/lock/artifact sinh bởi policy cân bằng grade trước P0 được giữ nguyên để bảo toàn provenance nhưng không còn hợp lệ cho kết luận xác nhận.

Protocol revision `three-signal-cuda-tie-aware-lf-2026-08-25` dùng threshold policy `negative-global-positive-tail-v1`: ngưỡng âm lấy phân vị 25 trên active development-train, còn ngưỡng dương lấy phân vị 75 trên phần strictly-above-negative. Cách này xử lý khối giá trị 0 của sparse skill overlap mà không dùng Gold-validation để học threshold. Artifact weak-supervision ghi policy, percentiles, thresholds và số quan sát trong từng tail; run cũ dùng semantics LF khác không được trộn hoặc diễn giải như run mới.

Các giả định Dawid–Skene được đặt tên và khóa trong cấu hình: số vòng EM, khoảng clip sensitivity/specificity, khoảng clip prior, accuracy khởi tạo và convergence tolerance. Artifact weak-supervision ghi cả các giả định này lẫn tham số ước lượng; xác suất đầu ra vẫn chỉ là nhãn yếu ước lượng, không phải ground truth.

Quy mô 400.000 pair cần nhiều RAM/CPU hơn bản cũ. Feature extraction đã được vector hóa theo entity để không gọi TF–IDF cho từng pair. Thời gian thực tế phụ thuộc phần cứng; nên chạy kernel sạch, đóng các tiến trình chiếm RAM và dành timeout tối thiểu hai giờ cho notebook full.

## Kiểm thử

```bash
python -m unittest discover -s tests -p "test_*_3signal.py" -v
```

Các test khóa schema dữ liệu, identity Gold, split query-disjoint, train-only preprocessing, feature contract ba tín hiệu, Dawid–Skene abstain-aware, pair delta, ListNet, metric theo query, paired bootstrap và one-time Gold-test gate.

## Phạm vi kết quả

Các số lịch sử của pipeline năm tín hiệu không phải kết quả xác nhận cho pipeline này. Kết luận chính chỉ được lấy từ full run ba tín hiệu khi paired-bootstrap CI 95% của chênh lệch nDCG@5 giữa LTR được chọn và `Score_H` không chứa 0.
