"""Testes dos endpoints. Requerem fastapi, httpx e sqlalchemy (pulados se ausentes).

Dois cenários reais: sem pesos (503) e com detector substituto no lugar do YOLO (contrato 200/400).
"""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from fastapi.testclient import TestClient

    from app.api.app_factory import create_app
    from app.application.analyze_image import AnalyzeImage
    from app.application.rupture_report import RuptureReport
    from app.application.track_rupture import TrackRupture
    from app.config import Settings
    from app.container import Container, build_container
    from app.domain.flow import FlowThresholds
    from app.infrastructure.file_store import FileImageStore
    from app.infrastructure.opencv_codec import OpenCvCodec
    from app.infrastructure.sql_repositories import (
        SqlEventRepository,
        SqlShiftRepository,
        create_session_factory,
        create_sqlite_engine,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    TestClient = None
    SKIP_REASON = f"dependência ausente: {exc.name}"
else:
    SKIP_REASON = ""

from tests.fakes import FakeDetector, StaticProfiles, cafe_profile, detection


def jpeg_bytes() -> bytes:
    ok, buffer = cv2.imencode(".jpg", np.full((48, 64, 3), 100, dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def upload(content=None, content_type="image/jpeg", name="foto.jpg"):
    return {"file": (name, jpeg_bytes() if content is None else content, content_type)}


@unittest.skipIf(TestClient is None, SKIP_REASON)
class ApiWithoutWeightsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        settings = Settings(
            model_path=root / "ausente.pt",
            person_model_path=root / "ausente.pt",
            data_dir=root / "data",
            received_dir=root / "received",
            results_dir=root / "results",
            shelf_config_path=root / "sem_arquivo.yaml",
        )
        client = TestClient(create_app(settings=settings))
        self.client = client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)

    def test_health_reports_model_not_loaded(self):
        body = self.client.get("/").json()
        self.assertEqual(body["status"], "ok")
        self.assertFalse(body["model"]["loaded"])
        self.assertIn("não encontrado", body["model"]["error"])

    def test_analyze_returns_503_with_clear_message(self):
        response = self.client.post("/analyze", files=upload())
        self.assertEqual(response.status_code, 503)
        self.assertIn("SDP_MODEL_PATH", response.json()["detail"])


@unittest.skipIf(TestClient is None, SKIP_REASON)
class ApiContractTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        engine = create_sqlite_engine(f"sqlite:///{root / 'api.db'}")
        self.addCleanup(engine.dispose)
        factory = create_session_factory(engine)
        events, shifts = SqlEventRepository(factory), SqlShiftRepository(factory)
        self.products = FakeDetector([detection("cafe", 0, 0.91234)], class_names={"cafe"})
        profiles = StaticProfiles(cafe_profile(capacity=10))
        settings = Settings(data_dir=root / "data", received_dir=root / "r", results_dir=root / "o")
        analyze = AnalyzeImage(
            products=self.products,
            people=FakeDetector([detection("person")], class_names={"person"}),
            codec=OpenCvCodec(),
            store=FileImageStore(root / "r", root / "o"),
            profiles=profiles,
            tracker=TrackRupture(events, shifts, confirmations_required=2),
            clock=datetime.now,
            max_bytes=settings.max_upload_bytes,
        )
        container = Container(
            settings=settings,
            analyze=analyze,
            report=RuptureReport(events, FlowThresholds(1.0, 3.0)),
            shifts=shifts,
            shelf_profiles=profiles,
            product_detector=self.products,
            people_detector=None,
        )
        container.load_models = lambda: None  # os detectores já são dublês prontos
        client = TestClient(create_app(container=container))
        self.client = client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)

    def test_analyze_success_matches_tcc_contract(self):
        response = self.client.post("/analyze", files=upload())
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["filename"], "foto.jpg")
        self.assertEqual(body["detections_count"], 1)
        item = body["detections"][0]
        self.assertEqual(set(item), {"class_id", "class_name", "confidence", "bbox"})
        self.assertEqual(set(item["bbox"]), {"x1", "y1", "x2", "y2"})
        self.assertEqual(item["confidence"], 0.9123)

    def test_unsupported_empty_and_corrupt_files_return_400_with_detail(self):
        cases = [
            upload(content_type="text/plain", name="a.txt"),
            upload(content=b""),
            upload(content=b"nao sou imagem"),
        ]
        for files in cases:
            with self.subTest(files=files["file"][2]):
                response = self.client.post("/analyze", files=files)
                self.assertEqual(response.status_code, 400)
                self.assertTrue(response.json()["detail"])

    def test_invalid_camera_id_returns_400(self):
        self.assertEqual(self.client.post("/analyze?camera_id=../x", files=upload()).status_code, 400)

    def test_missing_file_field_returns_422(self):
        self.assertEqual(self.client.post("/analyze").status_code, 422)

    def test_rupture_flow_through_api_and_csv_export(self):
        self.client.post("/analyze", files=upload())  # 1 detectado, mínimo 6 => pendente
        self.assertEqual(self.client.get("/ruptures").json(), [])
        self.client.post("/analyze", files=upload())  # confirma
        [open_event] = self.client.get("/ruptures?only_open=true").json()
        self.assertEqual(open_event["status"], "confirmada_em_andamento")

        self.products.detections = [detection("cafe")] * 8
        body = self.client.post("/analyze", files=upload()).json()
        self.assertEqual(body["shelf"]["products"][0]["state"], "disponivel")

        [closed] = self.client.get("/ruptures").json()
        self.assertEqual(closed["status"], "encerrada")
        csv_response = self.client.get("/ruptures/export.csv")
        self.assertEqual(csv_response.headers["content-type"].split(";")[0], "text/csv")
        self.assertIn("Tempo_Ruptura_Min", csv_response.text.splitlines()[0])

    def test_shift_crud(self):
        payload = {"employee": "Ana", "aisle": "C10", "starts_at": "2026-06-10T07:00:00", "ends_at": "2026-06-10T15:00:00"}
        created = self.client.post("/shifts", json=payload)
        self.assertEqual(created.status_code, 201)
        shift_id = created.json()["id"]
        self.assertEqual(len(self.client.get("/shifts").json()), 1)
        self.assertEqual(self.client.delete(f"/shifts/{shift_id}").status_code, 204)
        self.assertEqual(self.client.delete(f"/shifts/{shift_id}").status_code, 404)

    def test_shift_with_inverted_interval_is_rejected(self):
        payload = {"employee": "Ana", "aisle": "C10", "starts_at": "2026-06-10T15:00:00", "ends_at": "2026-06-10T07:00:00"}
        self.assertEqual(self.client.post("/shifts", json=payload).status_code, 422)


if __name__ == "__main__":
    unittest.main()
