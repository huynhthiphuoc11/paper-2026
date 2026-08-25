from __future__ import annotations

import ast
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks" / "experiments_3signal.ipynb"


def source_lines(text: str) -> list[str]:
    return [line + "\n" for line in text.strip("\n").split("\n")]


def markdown(cell_id: str, text: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source_lines(text)}


def code(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


def node_source(text: str, node: ast.AST) -> str:
    """Return a top-level node with decorators preserved verbatim."""
    first_line = node.lineno
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        first_line = min(first_line, *(item.lineno for item in decorators))
    lines = text.splitlines(keepends=True)
    return "".join(lines[first_line - 1:node.end_lineno]).rstrip()


def module_body(relative_path: str) -> str:
    """Return executable module code without package-local imports.

    The generated notebook contains the definitions themselves, so imports from
    src.experiments would be redundant and would make the notebook non-standalone.
    """
    path = ROOT / relative_path
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    kept = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__" or (node.module or "").startswith("src.experiments"):
                continue
        kept.append(node_source(text, node))
    return "\n\n\n".join(part for part in kept if part).strip() + "\n"


def definition_cells(prefix: str, text: str, max_lines: int = 70) -> list[dict]:
    """Place every top-level definition in an auditable executable cell."""
    tree = ast.parse(text)
    generated = []
    counter = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        counter += 1
        name = getattr(node, "name", f"block-{counter}")
        source = node_source(text, node)
        generated.append(code(f"{prefix}-{name}", source))
    return generated


def module_definition_cells(prefix: str, relative_path: str) -> list[dict]:
    return definition_cells(prefix, module_body(relative_path))


def standalone_runner() -> str:
    text = module_body("src/experiments/runner.py")
    old = '''def __init__(self, config_path: str | Path, smoke: bool = False):
        self.root = Path(__file__).resolve().parents[2]
        self.config_path = self._resolve(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.smoke = bool(smoke)
        self.settings = self._resolve_settings(self.config, self.smoke)
        self.device = _resolve_device(self.settings["models"]["device"])
        self.data_root = self._resolve(self.settings["data_dir"]).resolve()
        self.gold_path = self._resolve(self.settings["gold_path"]).resolve()
        annotation_value = self.settings["gold"].get("independent_annotations_path")
        self.independent_annotations_path = (
            self._resolve(annotation_value).resolve() if annotation_value else None
        )
        mode = "smoke" if self.smoke else "full"'''
    new = '''def __init__(self, config: dict, root: str | Path):
        self.root = Path(root).resolve()
        self.config_path = self.root / "configs" / "experiment_3signal.yaml"
        self.config = copy.deepcopy(config)
        self.smoke = False
        self.settings = copy.deepcopy(self.config)
        self.device = _resolve_device(self.settings["models"]["device"])
        self.data_root = self._resolve(self.settings["data_dir"]).resolve()
        self.gold_path = self._resolve(self.settings["gold_path"]).resolve()
        annotation_value = self.settings["gold"].get("independent_annotations_path")
        self.independent_annotations_path = (
            self._resolve(annotation_value).resolve() if annotation_value else None
        )
        mode = "full"'''
    if old not in text:
        raise RuntimeError("Runner constructor template changed; update notebook generator")
    text = text.replace(old, new)
    resolve_start = text.index("    @staticmethod\n    def _resolve_settings")
    resolve_end = text.index("\n\n    def _build_protocol_payload", resolve_start)
    text = text[:resolve_start] + text[resolve_end + 2:]
    text = text.replace("if not self.smoke and not label_condition_passed:", "if not label_condition_passed:")
    text = text.replace('"mode": "smoke" if self.smoke else "full",', '"mode": "full",')
    marker = "\n\n\ndef run_experiment("
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    return text


cells = [
    markdown("title", r"""
# Thực nghiệm weakly supervised Learning-to-Rank cho CV–Job Matching tiếng Việt

Notebook này là **tài liệu thực nghiệm chính, độc lập và có thể thực thi từ đầu đến cuối**. Toàn bộ mã nguồn cần thiết—từ audit dữ liệu, xây dựng ba tín hiệu, mô hình nhãn Dawid–Skene, ba hàm mục tiêu LTR, lựa chọn mô hình, khóa giao thức, đánh giá Gold-test đến paired bootstrap—được định nghĩa trực tiếp trong notebook; notebook không gọi logic thực nghiệm từ `src/`.

## Câu hỏi nghiên cứu

- **RQ chính:** Hàm xếp hạng học từ nhãn yếu có xếp hạng tốt hơn công thức điểm thủ công trên Gold-test độc lập hay không?
- **RQ1:** Xác suất ước lượng bởi mô hình nhãn có đạt điều kiện chất lượng đã khai báo trước trên Gold-validation hay không?
- **RQ2:** Pointwise, pairwise hay listwise phù hợp nhất khi giữ nguyên tập feature và năng lực scorer?

## Giao thức xác nhận

1. Mô hình đề xuất chỉ nhận ba tín hiệu: semantic, skill và experience.
2. Mọi phép fit preprocessing, ngưỡng labeling function và Dawid–Skene chỉ dùng development-train.
3. Gold được tách query-disjoint: số job validation khóa trong cấu hình, phần còn lại là Gold-test.
4. Gold-validation chỉ dùng cho ngưỡng posterior và lựa chọn formulation; không dùng Gold-test để tuning.
5. Gold-test chỉ được transform và đánh giá sau khi protocol đã khóa, đúng một lần trong một kernel sạch.
6. Đơn vị thống kê là job; metric được tính từng job rồi macro-average và bootstrap theo job.
7. Chỉ có hai ablation: thay label model bằng trung bình ba tín hiệu; bỏ LTR và xếp hạng trực tiếp bằng xác suất ước lượng.
8. Xác suất Dawid–Skene là **nhãn yếu ước lượng**, không phải ground truth. Gold hiện có là benchmark do tác giả gán nhãn, không phải outcome tuyển dụng.
"""),
    markdown("environment-md", r"""
## 1. Môi trường, khả năng tái lập và cấu hình khóa trước

Cell dưới đây tải thư viện, xác định thư mục dự án và đọc cấu hình full. Notebook **không có chế độ smoke**. Các seed, kích thước mẫu, hyperparameter grid, ngưỡng LF và số bootstrap được hiển thị trước khi xử lý Gold-test.
"""),
    code("environment", r'''
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import itertools
import json
import platform
import random
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn

def find_project_root(start: str | Path) -> Path:
    current = Path(start).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs" / "experiment_3signal.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find configs/experiment_3signal.yaml from the current directory"
    )


ROOT = find_project_root(Path.cwd())
CONFIG_PATH = ROOT / "configs" / "experiment_3signal.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
DATA_ROOT = (ROOT / CONFIG["data_dir"]).resolve()
GOLD_PATH = (ROOT / CONFIG["gold_path"]).resolve()
ANNOTATION_VALUE = CONFIG["gold"].get("independent_annotations_path")
INDEPENDENT_ANNOTATIONS_PATH = (
    (ROOT / ANNOTATION_VALUE).resolve() if ANNOTATION_VALUE else None
)

assert CONFIG["seeds"] == [11, 23, 42, 67, 89]
assert CONFIG["gold"]["validation_jobs"] == 4
assert CONFIG["gold"]["k_values"] == [5, 10]
assert CONFIG["sample"] == {
    "n_jobs": 2000, "n_candidates": 1500, "candidates_per_job": 200,
}
assert CONFIG["models"]["weight_decay"] == 1e-4
assert CONFIG["models"]["gradient_clip_norm"] == 5.0
REQUESTED_DEVICE = str(CONFIG["models"]["device"])
if REQUESTED_DEVICE.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(
        "The configured CUDA device is unavailable; the active PyTorch build "
        "does not expose an NVIDIA GPU"
    )
DEVICE = torch.device("cuda:0" if REQUESTED_DEVICE == "cuda" else REQUESTED_DEVICE)
assert CONFIG["weak_supervision"]["posterior_threshold"] == 0.5
assert CONFIG["weak_supervision"]["label_model"] == {
    "n_iter": 100,
    "parameter_clip": [0.51, 0.99],
    "prior_clip": [0.05, 0.95],
    "initial_accuracy": 0.75,
    "convergence_tolerance": 1e-7,
}
assert set(CONFIG["models"]["batch_sizes"]) == {
    "pointwise", "pairwise", "listwise",
}

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 140)
print("Project root:", ROOT)
print("Real data root:", DATA_ROOT)
print("Gold path:", GOLD_PATH)
print("Protocol version:", CONFIG["protocol_version"])
print("Configured device:", DEVICE)
if DEVICE.type == "cuda":
    print("CUDA device:", torch.cuda.get_device_name(DEVICE))
    print("CUDA runtime:", torch.version.cuda)
display(pd.DataFrame({
    "setting": ["seeds", "sample", "learning_rates", "batch_sizes", "epochs", "bootstrap_resamples"],
    "value": [
        str(CONFIG["seeds"]), str(CONFIG["sample"]),
        str(CONFIG["models"]["learning_rates"]), str(CONFIG["models"]["batch_sizes"]),
        CONFIG["models"]["max_epochs"], CONFIG["bootstrap"]["n_resamples"],
    ],
}))
'''),
    markdown("utilities-md", "## 2. Hằng số và tiện ích tái lập\n\nCác hàm băm, ghi artifact, chuẩn hóa Unicode và seed xác định được định nghĩa tại đây."),
    code("contracts", '''FEATURE_COLUMNS = ["s_sem", "s_skill", "s_exp"]\nLF_COLUMNS = ["lf_sem", "lf_skill", "lf_exp"]'''),
    *module_definition_cells("utils", "src/experiments/utils.py"),
    markdown("data-code-md", r"""
## 3. Nạp dữ liệu, kiểm tra identity và chia query

Identity ổn định được ánh xạ theo chỉ số dòng gốc (`JOB_<index>`, `CV_<index>`). CV trùng hoàn toàn được loại nhưng `_source_index` được bảo toàn. Manifest Gold dùng số job validation trong cấu hình, danh sách `job_id` đã sort và seed cố định; tuyệt đối không đọc grade/relevance khi chia split. IAA chỉ được tính từ file nhãn độc lập thật.
"""),
    *module_definition_cells("data", "src/experiments/data.py"),
    markdown("feature-code-md", r"""
## 4. Xây dựng ba tín hiệu và baseline thủ công

- `s_sem`: cosine giữa multilingual sentence embeddings của `Job Title` + `Job Description` + `Job Requirements` và `Desired Job` + `Target` + `Skills`.
- `s_skill`: Jaccard giữa tập kỹ năng đã chuẩn hóa.
- `s_exp`: độ tương thích giữa hai khoảng kinh nghiệm.

Model embedding pretrained được khóa trong cấu hình và không fine-tune bằng Gold. TF–IDF chỉ fit trên development-train cho role/description lexical components của baseline. Scorer học máy chỉ nhận đúng ba cột trên.
"""),
    *module_definition_cells("features", "src/experiments/features.py"),
    markdown("weak-code-md", r"""
## 5. Labeling functions và mô hình nhãn Dawid–Skene

Mỗi tín hiệu tạo một LF ba trạng thái `{−1, 0, +1}`. Ngưỡng 25/75 percentile chỉ fit trên development-train; `0` là abstain. EM ước lượng prior, sensitivity và specificity trong khi loại abstention khỏi mẫu số tương ứng. Các tham số bị đóng băng khi suy luận held-out.
"""),
    *module_definition_cells("weak", "src/experiments/weak_labels.py"),
    markdown("training-code-md", r"""
## 6. Pointwise, pairwise và listwise LTR

Ba formulation dùng cùng một linear scorer ba chiều:

- **Pointwise:** soft binary cross-entropy với posterior Dawid–Skene.
- **Pairwise:** RankNet trên cặp cùng job, với chênh lệch posterior cố định tối thiểu 0,02.
- **Listwise:** ListNet top-one cross-entropy theo query, với target là softmax của logit posterior đã clip.

Batch unit được khai báo riêng: CV–job pairs cho pointwise, preference pairs cho pairwise và whole queries cho listwise. Hyperparameter và early stopping tối đa hóa macro weak-validation nDCG@5. Gold-validation chọn formulation theo nDCG@5; các formulation nằm trong 0,005 của điểm cao nhất được so bằng common validation RankNet loss, rồi mới ưu tiên mô hình đơn giản hơn.
"""),
    *module_definition_cells("models", "src/experiments/models.py"),
    *module_definition_cells("training", "src/experiments/training.py"),
    markdown("evaluation-code-md", r"""
## 7. Metric, paired bootstrap và khóa Gold-test

nDCG@5, nDCG@10 và MRR được tính riêng cho từng job. MRR coi grade `≥2` là relevant. Paired bootstrap lấy mẫu lại các job, không coi từng cặp CV–job là quan sát độc lập.
"""),
    *module_definition_cells("evaluation", "src/experiments/evaluation.py"),
    *module_definition_cells("protocol", "src/experiments/protocol.py"),
    markdown("runner-code-md", r"""
## 8. Bộ điều phối giao thức full

Lớp dưới đây thực thi state machine: audit → development → weak supervision → formulation selection → chuẩn bị toàn bộ checkpoint chính/ablation → protocol lock → durable one-time Gold-test → finalize. Full run dừng trước ranking nếu điều kiện tiên quyết về chất lượng label model không đạt; đây là hành vi fail-fast theo giao thức, không phải lỗi kỹ thuật.
"""),
    *definition_cells("runner", standalone_runner()),
    markdown("real-data-md", r"""
## 9. Nguồn dữ liệu thực nghiệm thật và kiểm toán đầu vào

Thực nghiệm này chỉ đọc hai bảng raw tại `data/` tương đối từ project root và Gold benchmark đã khóa. Không có synthetic data, fixture hoặc fallback. Mỗi file được ghi absolute path đã resolve, dung lượng, thời điểm sửa và SHA-256 trước mọi phép lấy mẫu hay feature engineering.
"""),
    code("real-data-load", r'''
input_manifest = build_input_manifest(DATA_ROOT)
raw_preview = load_raw_data(DATA_ROOT)
gold_preview = load_gold_with_identity_check(GOLD_PATH, raw_preview)

display(pd.DataFrame(input_manifest).T.reset_index(names="input"))
display(pd.DataFrame({
    "dataset": ["Jobs raw", "Jobs clean", "CV raw", "CV clean", "Gold"],
    "rows": [
        raw_preview.audit["jobs_raw_rows"],
        raw_preview.audit["jobs_clean_rows"],
        raw_preview.audit["candidates_raw_rows"],
        raw_preview.audit["candidates_clean_rows"],
        len(gold_preview),
    ],
    "columns": [
        raw_preview.audit["jobs_columns"],
        raw_preview.jobs.shape[1] - 1,
        raw_preview.audit["candidates_columns"],
        raw_preview.candidates.shape[1] - 1,
        gold_preview.shape[1],
    ],
    "exact_duplicates_removed": [
        raw_preview.audit["jobs_exact_duplicates"], 0,
        raw_preview.audit["candidates_exact_duplicates"], 0, 0,
    ],
    "missing_cells_raw": [
        raw_preview.audit["jobs_missing_cells"], np.nan,
        raw_preview.audit["candidates_missing_cells"], np.nan, np.nan,
    ],
}))
display(gold_preview["relevance"].value_counts().sort_index().rename_axis("grade").to_frame("count"))

assert raw_preview.audit["data_root"] == str(DATA_ROOT)
assert raw_preview.audit["jobs_raw_rows"] == 14634
assert raw_preview.audit["jobs_clean_rows"] == 14634
assert raw_preview.audit["candidates_raw_rows"] == 3983
assert raw_preview.audit["candidates_exact_duplicates"] == 792
assert raw_preview.audit["candidates_clean_rows"] == 3191
assert len(gold_preview) == 100
assert gold_preview["job_id"].nunique() == 12
assert gold_preview["cand_id"].nunique() == 67
'''),
    markdown("run-audit-md", r"""
## 10. Audit identity và cố định Gold split

Runner đọc lại đúng các file đã băm ở trên, kiểm tra identity giữa Gold và dữ liệu gốc, grade hợp lệ và tính query-disjoint của split. Chưa có feature hoặc score Gold-test nào được tạo tại bước này.
"""),
    code("run-audit", r'''
experiment = ThreeSignalExperiment(CONFIG, ROOT)
gold_validation, raw_audit = experiment.audit_data_and_gold()

assert raw_audit["input_files"] == input_manifest
assert raw_audit["data_root"] == str(DATA_ROOT)
display(pd.DataFrame([{
    key: value for key, value in raw_audit.items()
    if key not in {"input_files", "jobs_required_columns", "candidates_required_columns"}
}]).T.rename(columns={0: "value"}))
display(pd.DataFrame({
    "partition": ["Gold-validation", "Gold-test"],
    "jobs": [len(experiment.gold_manifest["validation_jobs"]), len(experiment.gold_manifest["test_jobs"])],
    "pairs": [experiment.gold_manifest["validation_pairs"], experiment.gold_manifest["test_pairs"]],
}))
display(experiment.gold["relevance"].value_counts().sort_index().rename_axis("grade").to_frame("count"))

assert set(experiment.gold_manifest["validation_jobs"]).isdisjoint(experiment.gold_manifest["test_jobs"])
assert len(experiment.gold_manifest["validation_jobs"]) == CONFIG["gold"]["validation_jobs"]
assert len(experiment.gold_manifest["test_jobs"]) == experiment.gold["job_id"].nunique() - CONFIG["gold"]["validation_jobs"]
display(pd.DataFrame([experiment.iaa_audit]))
'''),
    markdown("run-development-md", r"""
## 11. Development pool thật, query-disjoint split và train-only preprocessing

Toàn bộ job và CV đã xuất hiện trong Gold đều bị loại khỏi development pool. Feature pipeline được fit duy nhất trên development-train; cùng trạng thái đó được dùng cho development-validation và Gold-validation.
"""),
    code("run-development", r'''
development_counts = experiment.prepare_development_data()
feature_manifest = experiment.feature_pipeline.manifest()

assert len(experiment.sampled.jobs) == 2_000
assert len(experiment.sampled.candidates) == 1_500
assert len(experiment.sampled.pairs) == 400_000
assert not experiment.sampled.pairs.duplicated(["job_id", "cand_id"]).any()
assert set(experiment.sampled.pairs["job_id"]).isdisjoint(set(experiment.gold["job_id"]))
assert set(experiment.sampled.pairs["cand_id"]).isdisjoint(set(experiment.gold["cand_id"]))
assert set(experiment.development_manifest["train_jobs"]).isdisjoint(
    experiment.development_manifest["validation_jobs"]
)
assert set(experiment.development_manifest["train_jobs"]).isdisjoint(
    experiment.development_manifest["test_jobs"]
)
assert set(experiment.development_manifest["validation_jobs"]).isdisjoint(
    experiment.development_manifest["test_jobs"]
)
assert set(experiment.sampled.pairs["job_source_index"]).issubset(
    set(raw_preview.jobs["_source_index"])
)
assert set(experiment.sampled.pairs["candidate_source_index"]).issubset(
    set(raw_preview.candidates["_source_index"])
)

display(pd.DataFrame([development_counts]).T.rename(columns={0: "count"}))
display(pd.DataFrame({
    "partition": ["train", "validation", "development-test"],
    "jobs": [
        len(experiment.development_manifest["train_jobs"]),
        len(experiment.development_manifest["validation_jobs"]),
        len(experiment.development_manifest["test_jobs"]),
    ],
    "pairs": [len(experiment.train), len(experiment.validation), len(experiment.development_test)],
}))
display(pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "train_missing_rate": [experiment.train_raw_features[c].isna().mean() for c in FEATURE_COLUMNS],
    "train_mean_after_imputation": [experiment.train[c].mean() for c in FEATURE_COLUMNS],
    "train_std_after_imputation": [experiment.train[c].std() for c in FEATURE_COLUMNS],
}))
display({
    "semantic_encoder": feature_manifest["semantic_encoder"],
    "role_lexical_vocabulary_size": feature_manifest["role_lexical_vocabulary_size"],
    "description_lexical_vocabulary_size": feature_manifest["description_lexical_vocabulary_size"],
    "feature_columns": feature_manifest["feature_columns"],
})

assert feature_manifest["feature_columns"] == FEATURE_COLUMNS
assert set(feature_manifest["fit_job_ids"]).isdisjoint(set(experiment.gold["job_id"]))
assert set(feature_manifest["fit_candidate_ids"]).isdisjoint(set(experiment.gold["cand_id"]))
'''),
    markdown("run-weak-md", r"""
## 11. Chất lượng nguồn nhãn trên Gold-validation

Đây là điều kiện tiên quyết của Core 1. Ngưỡng âm của mỗi LF lấy phân vị 25 trên development-train; để tránh sparse skill overlap bị suy biến do nhiều giá trị 0 trùng nhau, ngưỡng dương lấy phân vị 75 trên phần train strictly-above-negative. Các ngưỡng được đóng băng trước held-out inference. Luật strict 3/3 là baseline có abstention; Dawid–Skene được đánh giá tại threshold cố định 0,5, không calibration trên Gold-validation. Nếu điều kiện đã khai báo trước không đạt, full confirmatory run chủ động dừng và **không mở Gold-test**.
"""),
    code("run-weak", r'''
label_quality, lf_statistics, lf_pair_diagnostics = (
    experiment.fit_weak_supervision()
)

display(Markdown("### Thống kê labeling functions trên development-train"))
display(lf_statistics)
display(Markdown("### Phụ thuộc giữa các labeling functions"))
display(lf_pair_diagnostics)
display(Markdown("### Bảng chất lượng nhãn trên Gold-validation"))
display(label_quality)
print("Selected posterior threshold:", experiment.posterior_threshold)
print("Weak-supervision artifacts:", experiment.output_dir)

CONFIRMATORY_ALLOWED = bool(experiment.label_gate_passed)
if CONFIRMATORY_ALLOWED:
    display(Markdown(
        "**PASS:** label-model prerequisite đạt; các bước confirmatory được phép chạy."
    ))
else:
    strict_row = label_quality.loc[
        label_quality["method"] == "strict_3_of_3"
    ].iloc[0]
    label_row = label_quality.loc[
        label_quality["method"] == "dawid_skene"
    ].iloc[0]
    display(Markdown(
        "**BLOCKED:** label-model prerequisite không đạt. "
        f"Gold-validation có `{int(label_row['n_positive'])}` positive; "
        f"strict 3/3: TP={int(strict_row['tp'])}, FP={int(strict_row['fp'])}, "
        f"FN={int(strict_row['fn'])}, TN={int(strict_row['tn'])}; "
        f"Dawid–Skene: TP={int(label_row['tp'])}, FP={int(label_row['fp'])}, "
        f"FN={int(label_row['fn'])}, TN={int(label_row['tn'])}. "
        "Ranking confirmatory và Gold-test sẽ được bỏ qua; đây là kết quả "
        "fail-fast hợp lệ, không phải lỗi thực thi."
    ))
'''),
    markdown("run-formulation-md", r"""
## 12. Huấn luyện đa seed và lựa chọn formulation

Mỗi formulation được huấn luyện với năm seed và cùng hyperparameter grid. Hyperparameter được chọn bằng objective trên development-validation; formulation được chọn bằng macro nDCG@5 trên bốn Gold-validation jobs. Không có Gold-test score tại bước này.
"""),
    code("run-formulation", r'''
if CONFIRMATORY_ALLOWED:
    formulation_summary = experiment.train_and_select_formulation()
    display(formulation_summary.sort_values(
        "ndcg@5", ascending=False
    ).reset_index(drop=True))
    display(Markdown("### Chẩn đoán overfitting tại checkpoint tốt nhất"))
    display(experiment.overfitting_diagnostics.sort_values(
        ["formulation", "seed", "best_validation_loss"]
    ).reset_index(drop=True))
    print("Selected formulation:", experiment.selected_formulation)
else:
    formulation_summary = pd.DataFrame()
    display(Markdown(
        "**SKIPPED:** formulation training bị chặn bởi label-model gate."
    ))
'''),
    markdown("development-test-md", r"""
### 12.1 Final weak-ranking internal check

Development-test chỉ được dùng **sau khi** formulation và checkpoint đã được chọn. Kết quả này là diagnostic-only, không được dùng để thay đổi bất kỳ quyết định huấn luyện hay protocol nào.
"""),
    code("development-test", r'''
if CONFIRMATORY_ALLOWED:
    selected_before_development_test = experiment.selected_formulation
    development_test_diagnostic = experiment.evaluate_development_test_once()
    display(development_test_diagnostic)
    assert experiment.selected_formulation == selected_before_development_test
    assert set(development_test_diagnostic["selection_role"]) == {
        "diagnostic_only_after_selection"
    }
else:
    development_test_diagnostic = pd.DataFrame()
    display(Markdown(
        "**SKIPPED:** development-test diagnostic bị chặn bởi label-model gate."
    ))
'''),
    markdown("learning-curves-md", r"""
### 12.2 Learning curves và generalization gap

Đường train/validation objective được ghi ở mọi epoch cho checkpoint hyperparameter được chọn. Early stopping dùng macro weak-validation nDCG@5; Gold không tham gia quyết định dừng. Objective loss và khoảng cách validation–train là diagnostic, không phải tiêu chí chọn checkpoint.
"""),
    code("learning-curves", r'''
if CONFIRMATORY_ALLOWED:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
    for axis, formulation in zip(
        axes, ["pointwise", "pairwise", "listwise"]
    ):
        subset = experiment.training_history[
            experiment.training_history["formulation"] == formulation
        ]
        mean_curve = subset.groupby("epoch", as_index=False)[
            ["train_loss", "validation_loss"]
        ].mean()
        axis.plot(
            mean_curve["epoch"], mean_curve["train_loss"], label="train"
        )
        axis.plot(
            mean_curve["epoch"], mean_curve["validation_loss"],
            label="validation"
        )
        axis.set_title(formulation)
        axis.set_xlabel("epoch")
        axis.set_ylabel("objective loss")
        axis.grid(alpha=0.25)
        axis.legend()
    plt.suptitle("Mean learning curves across five seeds")
    plt.tight_layout()
    plt.show()
else:
    display(Markdown(
        "**SKIPPED:** không có learning curves vì ranker không được huấn luyện."
    ))
'''),
    markdown("run-lock-md", r"""
## 13. Chuẩn bị checkpoint và Protocol lock

Toàn bộ checkpoint của mô hình chính và ablation mean-signal được train, lưu và băm **trước** khi khóa. Threshold posterior cố định 0,5, formulation, seed, feature vocabulary hashes, tham số Dawid–Skene, development-test diagnostic-only, checkpoint metadata và hash Gold split được ghi ra trước khi Gold-test được transform. Cell assertion xác nhận test vẫn đóng tại thời điểm khóa.
"""),
    code("run-lock", r'''
if CONFIRMATORY_ALLOWED:
    checkpoint_metadata = experiment.prepare_confirmatory_checkpoints()
    assert set(checkpoint_metadata) == {"main", "ablation_mean_signal"}
    protocol_lock = experiment.lock_protocol()
    display(protocol_lock)
    assert protocol_lock["locked"] is True
    assert protocol_lock["test_opened"] is False
else:
    checkpoint_metadata = {}
    protocol_lock = {
        "locked": False,
        "test_opened": False,
        "status": "confirmatory_blocked",
        "reason": experiment.label_gate_reason,
    }
    display(protocol_lock)
'''),
    markdown("run-test-md", r"""
## 14. Đánh giá Gold-test đúng một lần

Cell này là điểm duy nhất mở Gold-test. Bốn hệ thống được báo cáo:

1. `manual_score_h`: công thức thủ công lịch sử.
2. `selected_ltr`: hệ thống đầy đủ.
3. `ablation_mean_signal_ltr`: thay Dawid–Skene bằng trung bình ba tín hiệu, giữ formulation.
4. `ablation_direct_probability`: bỏ LTR, xếp hạng trực tiếp bằng posterior Dawid–Skene.

Mỗi score được đánh giá theo job, trung bình qua seed trước khi bootstrap, rồi macro-average qua job.
"""),
    code("run-test", r'''
if CONFIRMATORY_ALLOWED:
    gold_test_summary, bootstrap_results, gold_test_per_job = (
        experiment.evaluate_gold_test_once()
    )
    display(Markdown("### Kết quả chính và đúng hai ablation"))
    display(gold_test_summary.sort_values(
        "ndcg@5", ascending=False
    ).reset_index(drop=True))
    display(Markdown("### Paired bootstrap 95% CI theo job"))
    display(bootstrap_results)
    assert set(gold_test_summary["system"]) == {
        "manual_score_h", "selected_ltr",
        "ablation_mean_signal_ltr", "ablation_direct_probability",
    }
    assert set(bootstrap_results["comparison"]) == {
        "main", "core_1", "core_2"
    }
else:
    gold_test_summary = pd.DataFrame()
    bootstrap_results = pd.DataFrame()
    gold_test_per_job = pd.DataFrame()
    display(Markdown(
        "**NOT OPENED:** Gold-test vẫn được giữ kín vì label-model gate thất bại."
    ))
    assert experiment.protocol.test_opened is False
'''),
    markdown("run-final-md", r"""
## 15. Kết luận xác nhận và artifact

Giả thuyết chính chỉ được ủng hộ khi cận dưới CI 95% của chênh lệch nDCG@5 (`selected_ltr − manual_score_h`) lớn hơn 0. Kết luận được sinh bằng quy tắc cố định, không lựa chọn diễn giải sau khi xem kết quả.
"""),
    code("run-final", r'''
if CONFIRMATORY_ALLOWED:
    run_manifest = experiment.finalize()
    main_ndcg5 = bootstrap_results.query(
        "comparison == 'main' and metric == 'ndcg@5'"
    ).iloc[0]
    display(Markdown(f"## {run_manifest['conclusion']}"))
    display(pd.DataFrame([main_ndcg5]))
    print("Authoritative artifacts:", experiment.output_dir)
    print("Protocol final state:", experiment.protocol.manifest())
else:
    run_manifest = {
        "conclusion": "CONFIRMATORY EXPERIMENT BLOCKED",
        "reason": experiment.label_gate_reason,
        "gold_test_opened": False,
        "artifact_directory": str(experiment.output_dir),
    }
    write_json(
        experiment.output_dir / "audit" / "blocked_run_manifest.json",
        run_manifest,
    )
    display(Markdown("## CONFIRMATORY EXPERIMENT BLOCKED"))
    display(pd.DataFrame([run_manifest]))
    print("Diagnostic artifacts:", experiment.output_dir)
    assert experiment.protocol.test_opened is False
'''),
    markdown("limitations", r"""
## 16. Giới hạn diễn giải

- Gold hiện chỉ gồm 12 query và được gán nhãn bởi một annotator; IAA phải báo `not_available` cho đến khi có file nhãn độc lập thật. Độ bất định có thể lớn và benchmark không đại diện cho quyết định tuyển dụng thực tế.
- Dataset không có click, apply, interview hoặc hiring outcome.
- Dawid–Skene giả định độc lập có điều kiện, trong khi embedding text có thể chứa thông tin về skill và experience; tương quan LF được báo cáo như diagnostic chứ không chứng minh giả định đúng.
- Ba tín hiệu là proxy quan sát được. Posterior của label model chỉ là xác suất ước lượng dùng để huấn luyện, không phải ground truth.
- Với Gold hiện tại, Gold-test chỉ có tám job; paired-bootstrap CI phải được báo cáo cùng point estimate và không suy rộng từ từng cặp CV–job như các quan sát độc lập.
- Nếu điều kiện chất lượng label model thất bại ở Mục 11, nghiên cứu không được phép bỏ qua gate, điều chỉnh theo Gold-test hoặc diễn giải kết quả ranking xác nhận.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
