"""Integração com o YOLO11 real. Requer ultralytics e pesos em models/ (pulado se ausentes).

Não valida acurácia de produtos: apenas que o adaptador carrega pesos reais e devolve o formato esperado.
"""
import unittest
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "yolo11n.pt"

try:
    import ultralytics  # noqa: F401
except ModuleNotFoundError:
    HAS_ULTRALYTICS = False
else:
    HAS_ULTRALYTICS = True

from app.infrastructure.yolo_detector import YoloDetector


@unittest.skipUnless(HAS_ULTRALYTICS and WEIGHTS.is_file(), "requer ultralytics e models/yolo11n.pt")
class YoloIntegrationTest(unittest.TestCase):
    def test_loads_weights_and_exposes_coco_classes(self):
        detector = YoloDetector(WEIGHTS, confidence=0.25, imgsz=640)
        detector.load()
        self.assertTrue(detector.is_available, detector.load_error)
        self.assertIn("person", detector.class_names)

    def test_detect_returns_list_for_blank_image(self):
        detector = YoloDetector(WEIGHTS, confidence=0.25, imgsz=640, device="cpu")
        detector.load()
        self.assertEqual(detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)), [])

    def test_class_filter_restricts_class_names(self):
        detector = YoloDetector(WEIGHTS, confidence=0.25, imgsz=640, class_ids=[0])
        detector.load()
        self.assertEqual(detector.class_names, frozenset({"person"}))


if __name__ == "__main__":
    unittest.main()
