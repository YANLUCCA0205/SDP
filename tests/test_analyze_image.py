import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

from app.application.analyze_image import AnalyzeImage, AnalyzeRequest
from app.application.errors import InvalidImageError, ModelUnavailableError
from app.application.track_rupture import TrackRupture
from app.infrastructure.file_store import FileImageStore
from app.infrastructure.opencv_codec import OpenCvCodec
from tests.fakes import (
    T0,
    FakeDetector,
    InMemoryEventRepository,
    InMemoryShiftRepository,
    StaticProfiles,
    cafe_profile,
    detection,
)

MAX_BYTES = 200_000


def encoded_image(extension: str = ".jpg") -> bytes:
    pixels = np.full((48, 64, 3), 127, dtype=np.uint8)
    ok, buffer = cv2.imencode(extension, pixels)
    assert ok
    return buffer.tobytes()


class AnalyzeImageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.events = InMemoryEventRepository()
        self.products = FakeDetector([detection("cafe")], class_names={"cafe", "arroz"})
        self.people = FakeDetector([detection("person"), detection("person")], class_names={"person"})
        self.clock_now = T0
        self.build(profiles=StaticProfiles(cafe_profile(capacity=10)))

    def build(self, *, profiles, products=None, people="default"):
        self.use_case = AnalyzeImage(
            products=products or self.products,
            people=self.people if people == "default" else people,
            codec=OpenCvCodec(),
            store=FileImageStore(self.root / "received", self.root / "results"),
            profiles=profiles,
            tracker=TrackRupture(self.events, InMemoryShiftRepository(), confirmations_required=2),
            clock=lambda: self.clock_now,
            max_bytes=MAX_BYTES,
        )

    def request(self, **overrides) -> AnalyzeRequest:
        values = dict(camera_id="cam-01", filename="x.jpg", content_type="image/jpeg", data=encoded_image())
        values.update(overrides)
        return AnalyzeRequest(**values)

    # --- validação de entrada (contrato 400 do TCC) ---
    def test_rejects_unsupported_content_type(self):
        with self.assertRaisesRegex(InvalidImageError, "não suportado"):
            self.use_case.execute(self.request(content_type="image/gif"))

    def test_rejects_empty_file(self):
        with self.assertRaisesRegex(InvalidImageError, "vazio"):
            self.use_case.execute(self.request(data=b""))

    def test_rejects_undecodable_image(self):
        with self.assertRaisesRegex(InvalidImageError, "decodificar"):
            self.use_case.execute(self.request(data=b"isto nao e uma imagem"))

    def test_rejects_file_over_size_limit(self):
        with self.assertRaisesRegex(InvalidImageError, "limite"):
            self.use_case.execute(self.request(data=b"0" * (MAX_BYTES + 1)))

    def test_rejects_camera_id_that_could_escape_the_storage_directory(self):
        for bad in ("../etc", "a/b", "", "x" * 65, "cam 01"):
            with self.subTest(camera_id=bad), self.assertRaisesRegex(InvalidImageError, "camera_id"):
                self.use_case.execute(self.request(camera_id=bad))
        self.assertFalse((self.root / "received").exists())

    # --- modelo ausente ---
    def test_model_unavailable_raises_and_stores_nothing(self):
        self.build(profiles=StaticProfiles(), products=FakeDetector(available=False))
        with self.assertRaises(ModelUnavailableError):
            self.use_case.execute(self.request())
        self.assertFalse((self.root / "received").exists())

    # --- fluxo feliz ---
    def test_returns_detections_and_people_count(self):
        result = self.use_case.execute(self.request())
        self.assertEqual([d.class_name for d in result.detections], ["cafe"])
        self.assertEqual(result.people_count, 2)
        self.assertEqual(result.camera_id, "cam-01")

    def test_stores_unique_history_and_latest_copies_per_camera(self):
        self.use_case.execute(self.request())
        self.clock_now = T0 + timedelta(seconds=30)
        self.use_case.execute(self.request())
        received = self.root / "received" / "cam-01"
        results = self.root / "results" / "cam-01"
        self.assertEqual(len([p for p in received.glob("2026*.jpg")]), 2)
        self.assertEqual(len([p for p in results.glob("2026*.jpg")]), 2)
        self.assertTrue((received / "ultima_imagem.jpg").is_file())
        self.assertTrue((results / "ultima_deteccao.jpg").is_file())
        annotated = cv2.imread(str(results / "ultima_deteccao.jpg"))
        self.assertEqual(annotated.shape, (48, 64, 3))

    def test_png_is_stored_with_png_extension(self):
        self.use_case.execute(self.request(content_type="image/png", data=encoded_image(".png")))
        self.assertTrue(list((self.root / "received" / "cam-01").glob("*.png")))

    def test_cameras_are_stored_separately(self):
        self.use_case.execute(self.request(camera_id="cam-01"))
        self.use_case.execute(self.request(camera_id="cam-02"))
        self.assertTrue((self.root / "received" / "cam-01").is_dir())
        self.assertTrue((self.root / "received" / "cam-02").is_dir())

    # --- regra de ruptura ---
    def test_without_profile_there_is_no_shelf_analysis(self):
        self.build(profiles=StaticProfiles())
        self.assertIsNone(self.use_case.execute(self.request()).shelf)

    def test_rupture_is_registered_and_confirmed_on_second_capture(self):
        # 1 café detectado; mínimo esperado = 6 => possível ruptura.
        first = self.use_case.execute(self.request()).shelf
        reading = first.evaluation.readings[0]
        self.assertTrue(reading.in_rupture)
        self.assertEqual((reading.detected, reading.required), (1, 6))
        self.assertFalse(first.open_events["cafe"].confirmed)

        self.clock_now = T0 + timedelta(seconds=30)
        second = self.use_case.execute(self.request()).shelf
        self.assertTrue(second.open_events["cafe"].confirmed)
        self.assertEqual(len(self.events.list()), 1)

    def test_replenished_shelf_closes_the_event(self):
        self.use_case.execute(self.request())
        self.clock_now = T0 + timedelta(seconds=30)
        self.use_case.execute(self.request())
        self.products.detections = [detection("cafe")] * 7
        self.clock_now = T0 + timedelta(minutes=10)
        shelf = self.use_case.execute(self.request()).shelf
        self.assertIsNone(shelf.open_events["cafe"])
        [event] = self.events.list()
        self.assertEqual(event.duration_minutes, 10.0)

    def test_profile_products_unknown_to_the_model_are_not_flagged_as_rupture(self):
        # Modelo genérico (COCO) com perfil de produtos reais: não pode gerar ruptura falsa.
        self.build(profiles=StaticProfiles(cafe_profile()), products=FakeDetector([detection("car")], {"car"}))
        shelf = self.use_case.execute(self.request()).shelf
        self.assertEqual(shelf.evaluation.unmonitored, ("cafe",))
        self.assertEqual(self.events.list(confirmed_only=False), [])

    # --- fluxo de clientes ---
    def test_people_count_is_none_when_person_model_is_unavailable(self):
        self.build(profiles=StaticProfiles(), people=FakeDetector(available=False))
        self.assertIsNone(self.use_case.execute(self.request()).people_count)

    def test_people_count_is_none_when_counting_is_disabled(self):
        self.build(profiles=StaticProfiles(), people=None)
        self.assertIsNone(self.use_case.execute(self.request()).people_count)

    def test_person_detector_failure_does_not_break_product_analysis(self):
        self.build(profiles=StaticProfiles(), people=FakeDetector(fail=True))
        with self.assertLogs("app.application.analyze_image", level="ERROR"):
            result = self.use_case.execute(self.request())
        self.assertIsNone(result.people_count)
        self.assertEqual(len(result.detections), 1)


if __name__ == "__main__":
    unittest.main()
