import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "training" / "check_dataset.py"
spec = importlib.util.spec_from_file_location("check_dataset", SCRIPT)
check_dataset = importlib.util.module_from_spec(spec)
sys.modules["check_dataset"] = check_dataset  # exigido por @dataclass com `from __future__ import annotations`
spec.loader.exec_module(check_dataset)

DATA_YAML = "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: a\n  1: b\n"


class CheckDatasetTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "data.yaml").write_text(DATA_YAML, encoding="utf-8")

    def add(self, split: str, name: str, label: str | None):
        images = self.root / "images" / split
        labels = self.root / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        (images / f"{name}.jpg").write_bytes(b"x")
        if label is not None:
            (labels / f"{name}.txt").write_text(label, encoding="utf-8")

    def run_check(self, splits=("train", "val", "test")):
        return check_dataset.check_dataset(self.root / "data.yaml", splits)

    def valid_dataset(self):
        for split in ("train", "val", "test"):
            self.add(split, "img1", "0 0.5 0.5 0.2 0.3\n1 0.2 0.2 0.1 0.1\n")

    def test_valid_dataset_passes_and_counts_instances(self):
        self.valid_dataset()
        report, _ = self.run_check()
        self.assertTrue(report.ok, report.problems)
        self.assertEqual(report.images, {"train": 1, "val": 1, "test": 1})
        self.assertEqual(report.instances, {0: 3, 1: 3})

    def test_empty_label_file_is_a_valid_background_image(self):
        self.valid_dataset()
        self.add("train", "fundo", "")
        self.assertTrue(self.run_check()[0].ok)

    def test_detects_missing_label(self):
        self.valid_dataset()
        self.add("val", "sem_rotulo", None)
        self.assertTrue(any("rótulo ausente" in p for p in self.run_check()[0].problems))

    def test_detects_orphan_label(self):
        self.valid_dataset()
        (self.root / "labels" / "train" / "orfao.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        self.assertTrue(any("sem imagem" in p for p in self.run_check()[0].problems))

    def test_detects_invalid_label_lines(self):
        cases = {
            "classe": "5 0.5 0.5 0.1 0.1",
            "normalizadas": "0 1.5 0.5 0.1 0.1",
            "campos": "0 0.5 0.5 0.1",
            "numéricos": "0 a b c d",
            "maiores que zero": "0 0.5 0.5 0 0.1",
        }
        for expected, line in cases.items():
            with self.subTest(expected=expected):
                self.assertIn(expected, check_dataset.check_label_line(line, 2))

    def test_reports_split_without_images(self):
        report, _ = self.run_check(("train",))
        self.assertTrue(any("nenhuma imagem" in p for p in report.problems))

    def test_shipped_data_yaml_is_well_formed(self):
        shipped = SCRIPT.parent.parent / "dataset" / "data.yaml"
        report, names = check_dataset.check_dataset(shipped, ())
        self.assertTrue(names)
        self.assertTrue(report.ok)


if __name__ == "__main__":
    unittest.main()
