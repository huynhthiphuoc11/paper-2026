import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "experiments_3signal.ipynb"


class NotebookStructureTests(unittest.TestCase):
    @classmethod
    def notebook_source(cls):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        return "\n\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_notebook_classes_are_cohesive_and_not_monkey_patched(self):
        source = self.notebook_source()
        tree = ast.parse(source)
        class_methods = {}
        monkey_patches = []
        generated_methods = []
        empty_classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [
                    item.name for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                class_methods[node.name] = methods
                if not methods and all(isinstance(item, ast.Pass) for item in node.body):
                    empty_classes.append(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in class_methods
                    ):
                        monkey_patches.append(f"{target.value.id}.{target.attr}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith((
                    "_ThreeSignalFeaturePipeline_",
                    "_DawidSkeneThreeSource_",
                    "_ThreeSignalExperiment_",
                )):
                    generated_methods.append(node.name)
        self.assertEqual(empty_classes, [])
        self.assertEqual(monkey_patches, [])
        self.assertEqual(generated_methods, [])
        self.assertIn("fit", class_methods["ThreeSignalFeaturePipeline"])
        self.assertIn("predict_proba", class_methods["DawidSkeneThreeSource"])
        self.assertIn("lock_protocol", class_methods["ThreeSignalExperiment"])
        decorated = {
            node.name: [ast.unparse(item) for item in node.decorator_list]
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        }
        for name in [
            "RawData", "SampledData", "FeatureState", "PreparedDataset",
            "PairwiseState", "TrainingResult", "ExperimentProtocol",
        ]:
            self.assertIn("dataclass", decorated[name])

    def test_notebook_remains_standalone_and_parseable(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for index, cell in enumerate(code_cells, 1):
            ast.parse("".join(cell.get("source", [])), filename=f"cell-{index}")
        source = self.notebook_source()
        self.assertNotIn("src.experiments", source)
        self.assertNotIn("SMOKE", source)
        self.assertIn("environment_manifest.json", source)
        self.assertIn('"assumptions"', source)
        self.assertNotIn("select_posterior_threshold", source)
        self.assertEqual(source.count("Job Description") >= 1, True)
        self.assertIn('CONFIG["models"]["device"]', source)
        self.assertIn("torch.cuda.is_available()", source)
        self.assertIn('DATA_ROOT = (ROOT / CONFIG["data_dir"]).resolve()', source)
        self.assertIn('def find_project_root', source)
        self.assertNotIn('ROOT_CANDIDATES', source)
        self.assertNotIn(r'D:\\NCKH\\paper-2026', source)
        self.assertIn('SentenceTransformer', source)
        self.assertIn('inter_annotator_agreement.json', source)
        self.assertIn("negative-global-positive-tail-v1", source)
        self.assertIn("gold_validation_weak_label_diagnostics.csv", source)
        self.assertIn("CONFIRMATORY_ALLOWED = bool(experiment.label_gate_passed)", source)
        self.assertIn("experiment.protocol.test_opened is False", source)
        self.assertIn("load_raw_data(DATA_ROOT)", source)
        self.assertIn("build_input_manifest(DATA_ROOT)", source)
        self.assertIn('raw_preview.audit["jobs_raw_rows"] == 14634', source)
        self.assertIn('len(experiment.sampled.pairs) == 400_000', source)


if __name__ == "__main__":
    unittest.main()
