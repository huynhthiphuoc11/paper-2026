import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.experiments.data import (
    build_input_manifest,
    inter_annotator_agreement,
    load_gold_with_identity_check,
    load_raw_data,
    make_gold_split_manifest,
    resolve_data_paths,
)


ROOT = Path(__file__).resolve().parents[1]


class DataProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_raw_data(ROOT / "data")
        cls.gold = load_gold_with_identity_check(
            ROOT / "data" / "gold" / "human_validated_benchmark_graded_0_3.csv",
            cls.raw,
        )

    def test_raw_audit_matches_dataset(self):
        self.assertEqual(self.raw.audit["jobs_raw_rows"], 14634)
        self.assertEqual(self.raw.audit["jobs_columns"], 19)
        self.assertEqual(self.raw.audit["candidates_raw_rows"], 3983)
        self.assertEqual(self.raw.audit["candidates_columns"], 14)
        self.assertEqual(self.raw.audit["candidates_exact_duplicates"], 792)
        self.assertEqual(self.raw.audit["candidates_unique"], 3191)

    def test_gold_identity_and_distribution(self):
        self.assertEqual(len(self.gold), 100)
        self.assertEqual(self.gold["job_id"].nunique(), 12)
        self.assertEqual(self.gold["cand_id"].nunique(), 67)
        self.assertEqual(self.gold["relevance"].value_counts().sort_index().to_dict(), {0: 30, 1: 35, 2: 29, 3: 6})

    def test_gold_split_is_deterministic_four_eight(self):
        first = make_gold_split_manifest(self.gold, 4, 42)
        second = make_gold_split_manifest(self.gold, 4, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first["validation_jobs"]), 4)
        self.assertEqual(len(first["test_jobs"]), 8)
        self.assertTrue(set(first["validation_jobs"]).isdisjoint(first["test_jobs"]))

    def test_gold_split_does_not_read_relevance(self):
        original = make_gold_split_manifest(self.gold, 4, 42)
        altered = self.gold.copy()
        altered["relevance"] = altered["relevance"].sample(
            frac=1.0, random_state=19
        ).to_numpy()
        shuffled = make_gold_split_manifest(altered, 4, 42)
        self.assertEqual(original, shuffled)

    def test_gold_split_supports_more_than_twelve_jobs(self):
        expanded = pd.concat([
            self.gold,
            self.gold.iloc[[0]].assign(job_id="JOB_999"),
        ], ignore_index=True)
        manifest = make_gold_split_manifest(expanded, 4, 42)
        self.assertEqual(manifest["total_jobs"], 13)
        self.assertEqual(len(manifest["validation_jobs"]), 4)
        self.assertEqual(len(manifest["test_jobs"]), 9)

    def test_missing_independent_annotations_is_explicit(self):
        audit = inter_annotator_agreement(None, self.gold)
        self.assertEqual(audit["status"], "not_available")

    def test_inter_annotator_agreement_uses_real_overlap(self):
        pairs = self.gold.iloc[:4][["job_id", "cand_id"]]
        annotations = pd.concat([
            pairs.assign(annotator_id="a", relevance=[0, 1, 2, 3]),
            pairs.assign(annotator_id="b", relevance=[0, 1, 2, 3]),
        ], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.csv"
            annotations.to_csv(path, index=False)
            audit = inter_annotator_agreement(path, self.gold)
        self.assertEqual(audit["status"], "available")
        self.assertEqual(audit["overlap_pairs"], 4)
        self.assertEqual(audit["exact_agreement"], 1.0)
        self.assertEqual(audit["quadratic_weighted_kappa"], 1.0)

    def test_inter_annotator_agreement_rejects_duplicate(self):
        pair = self.gold.iloc[[0]][["job_id", "cand_id"]]
        annotations = pd.concat([
            pair.assign(annotator_id="a", relevance=2),
            pair.assign(annotator_id="a", relevance=2),
            pair.assign(annotator_id="b", relevance=2),
        ], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.csv"
            annotations.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate annotation"):
                inter_annotator_agreement(path, self.gold)

    def test_real_input_paths_and_manifest_are_explicit(self):
        paths = resolve_data_paths(ROOT / "data")
        self.assertEqual(paths["jobs"], (ROOT / "data" / "JOB_DATA_FINAL.csv").resolve())
        self.assertEqual(paths["candidates"], (ROOT / "data" / "USER_DATA_FINAL.csv").resolve())
        manifest = build_input_manifest(ROOT / "data")
        self.assertEqual(set(manifest), {"jobs", "candidates"})
        for record in manifest.values():
            self.assertTrue(Path(record["path"]).is_absolute())
            self.assertGreater(record["size_bytes"], 0)
            self.assertEqual(len(record["sha256"]), 64)
            self.assertIn("modified_at_utc", record)

    def test_missing_real_data_files_fail_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "JOB_DATA_FINAL.csv"):
                resolve_data_paths(directory)

    def test_raw_audit_records_real_sources_and_clean_rows(self):
        audit = self.raw.audit
        self.assertEqual(audit["candidates_clean_rows"], 3191)
        self.assertEqual(audit["jobs_clean_rows"], 14634)
        self.assertEqual(audit["data_root"], str((ROOT / "data").resolve()))
        self.assertEqual(
            Path(audit["input_files"]["jobs"]["path"]),
            (ROOT / "data" / "JOB_DATA_FINAL.csv").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
