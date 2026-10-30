import csv
import io
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from app.application.rupture_report import RuptureReport
from app.domain.flow import FlowThresholds
from app.domain.rupture import RuptureEvent
from app.infrastructure.csv_export import COLUMNS, records_to_csv
from app.infrastructure.shelf_config import ConfigurationError, YamlShelfProfiles
from app.infrastructure.yolo_detector import YoloDetector
from tests.fakes import T0, InMemoryEventRepository


def closed_event(repo: InMemoryEventRepository, people: int | None) -> RuptureEvent:
    event = repo.add(
        RuptureEvent.open(
            camera_id="cam-01", aisle="C10", product="cafe", category="Mercearia",
            at=T0, people=people, confirmations_required=1,
        )
    )
    event.close(T0 + timedelta(minutes=47), "Ana")
    repo.update(event)
    return event


class RuptureReportTest(unittest.TestCase):
    def setUp(self):
        self.repo = InMemoryEventRepository()
        self.report = RuptureReport(self.repo, FlowThresholds(1.0, 3.0))

    def test_record_uses_tcc_columns_and_derives_flow_level(self):
        closed_event(self.repo, people=5)
        [record] = self.report.records()
        self.assertEqual(record["data"], "2026-06-10")
        self.assertEqual((record["h_inicio"], record["h_fim"]), ("08:00:00", "08:47:00"))
        self.assertEqual(record["tempo_ruptura_min"], 47.0)
        self.assertEqual(record["produto"], "cafe")
        self.assertEqual(record["categoria"], "Mercearia")
        self.assertEqual(record["corredor"], "C10")
        self.assertEqual(record["funcionario"], "Ana")
        self.assertEqual(record["fluxo_de_cliente"], "alto")
        self.assertEqual(record["status"], "encerrada")

    def test_flow_is_null_when_people_were_never_counted(self):
        closed_event(self.repo, people=None)
        self.assertIsNone(self.report.records()[0]["fluxo_de_cliente"])

    def test_open_event_has_no_end_or_duration(self):
        self.repo.add(
            RuptureEvent.open(camera_id="c", aisle="A", product="p", category="k", at=T0, people=0, confirmations_required=1)
        )
        [record] = self.report.records()
        self.assertEqual(record["status"], "confirmada_em_andamento")
        self.assertIsNone(record["h_fim"])
        self.assertIsNone(record["tempo_ruptura_min"])

    def test_pending_events_are_hidden_unless_requested(self):
        self.repo.add(
            RuptureEvent.open(camera_id="c", aisle="A", product="p", category="k", at=T0, people=None, confirmations_required=2)
        )
        self.assertEqual(self.report.records(), [])
        self.assertEqual(self.report.records(confirmed_only=False)[0]["status"], "pendente")


class CsvExportTest(unittest.TestCase):
    def test_exports_header_even_without_events(self):
        header = records_to_csv([]).strip().splitlines()[0]
        self.assertEqual(header.split(","), list(COLUMNS.values()))

    def test_exports_rows_with_tcc_column_names(self):
        repo = InMemoryEventRepository()
        closed_event(repo, people=0)
        rows = list(csv.DictReader(io.StringIO(records_to_csv(RuptureReport(repo, FlowThresholds(1, 3)).records()))))
        self.assertEqual(rows[0]["Produto"], "cafe")
        self.assertEqual(rows[0]["Tempo_Ruptura_Min"], "47.0")
        self.assertEqual(rows[0]["Fluxo_de_Cliente"], "baixo")


class YamlShelfProfilesTest(unittest.TestCase):
    def write(self, text: str) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "shelves.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_loads_profiles_per_camera(self):
        path = self.write(
            "cameras:\n"
            "  cam-01:\n    aisle: C10\n    products:\n      - {class_name: cafe, category: M, capacity: 10}\n"
            "  cam-02:\n    aisle: C11\n    products:\n      - {class_name: arroz, category: M, capacity: 5}\n"
        )
        profiles = YamlShelfProfiles.from_file(path, 0.6)
        self.assertEqual(profiles.camera_ids, ["cam-01", "cam-02"])
        self.assertEqual(profiles.get("cam-01").aisle, "C10")
        self.assertEqual(profiles.get("cam-01").slots[0].required_units(0.6), 6)
        self.assertIsNone(profiles.get("cam-99"))

    def test_missing_file_means_no_profiles(self):
        with self.assertLogs("app.infrastructure.shelf_config", level="WARNING"):
            profiles = YamlShelfProfiles.from_file(Path("/nonexistent/shelves.yaml"), 0.6)
        self.assertEqual(profiles.camera_ids, [])

    def test_invalid_content_raises_configuration_error(self):
        for text in (
            "cameras:\n  c:\n    aisle: A\n",  # sem products
            "cameras:\n  c:\n    aisle: A\n    products:\n      - {class_name: x, category: y, capacity: 0}\n",
            "cameras: [unbalanced",
        ):
            with self.subTest(text=text), self.assertRaises(ConfigurationError):
                YamlShelfProfiles.from_file(self.write(text), 0.6)

    def test_shipped_example_config_is_valid(self):
        example = Path(__file__).resolve().parent.parent / "config" / "shelves.yaml"
        profiles = YamlShelfProfiles.from_file(example, 0.6)
        self.assertTrue(profiles.camera_ids)


class YoloDetectorWithoutWeightsTest(unittest.TestCase):
    def test_missing_weights_leave_detector_unavailable_without_fallback(self):
        detector = YoloDetector(Path("/nonexistent/best.pt"), confidence=0.25, imgsz=640)
        with self.assertLogs("app.infrastructure.yolo_detector", level="WARNING"):
            detector.load()
        self.assertFalse(detector.is_available)
        self.assertEqual(detector.class_names, frozenset())
        self.assertIn("não encontrado", detector.load_error)

    def test_detect_without_loaded_model_fails_loudly(self):
        detector = YoloDetector(Path("/nonexistent/best.pt"), confidence=0.25, imgsz=640)
        with self.assertRaises(RuntimeError):
            detector.detect(None)


if __name__ == "__main__":
    unittest.main()
