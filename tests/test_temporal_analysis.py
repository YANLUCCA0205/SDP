"""Testes da integracao temporal com mascara e detector injetado."""
from pathlib import Path

import cv2
import numpy as np

from app.application.temporal_analysis import analyze_history


class FakeDetector:
    is_available = True

    def __init__(self):
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return []


def test_temporal_multishelf_with_yolo_hook(tmp_path: Path):
    root = tmp_path / "received"
    photo_dir = root / "cam-01" / "2026-11-03"
    photo_dir.mkdir(parents=True)
    original = np.full((480, 640, 3), 200, dtype=np.uint8)
    changed = original.copy()
    changed[284:346, 244:318] = (10, 10, 10)
    for prefix, image in (("221525", original), ("221557", changed), ("221802", changed)):
        assert cv2.imwrite(str(photo_dir / f"{prefix}_fixture.jpg"), image)
    config = tmp_path / "shelves.yaml"
    config.write_text("cameras:\n  cam-01:\n    terceira: [0.03, 0.56, 0.90, 0.76]\n",
                      encoding="utf-8")
    detector = FakeDetector()
    result = analyze_history(
        camera_id="cam-01", day="2026-11-03", reference_prefix="221525",
        received_root=root, results_root=tmp_path / "results",
        config_path=config, detector=detector,
    )
    assert result["processed_captures"] == 3
    assert [row["status"] for row in result["readings"]] == [
        "SEM_ALERTA", "SUSPEITA", "PERSISTENTE"
    ]
    assert detector.calls == 3
    assert Path(result["csv"]).is_file()
    assert len(list((tmp_path / "results" / "temporal" / "cam-01" / "2026-11-03").glob("*_mascara.png"))) == 3
