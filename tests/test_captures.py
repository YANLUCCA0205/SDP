"""Teste do endpoint de coleta histórica e isolamento da inferência."""
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.app_factory import create_app
from app.config import Settings
from app.infrastructure.capture_store import CaptureRow
from app.infrastructure.sql_repositories import create_session_factory, create_sqlite_engine


def test_capture_persists_without_model():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        settings = Settings(
            data_dir=root / "data", received_dir=root / "received",
            results_dir=root / "results", model_path=root / "no-model.pt",
            people_counting_enabled=False, shelf_config_path=root / "missing.yaml",
        )
        ok, encoded = cv2.imencode(".jpg", np.full((32, 48, 3), 127, dtype=np.uint8))
        assert ok
        with TestClient(create_app(settings=settings)) as client:
            response = client.post(
                "/captures?camera_id=cam-01",
                files={"file": ("test.jpg", encoded.tobytes(), "image/jpeg")},
            )
            assert response.status_code == 201, response.text
            result = response.json()
            assert result["status"] == "stored"
            assert result["processing_status"] == "PENDING"
            assert result["camera_id"] == "cam-01"
            db = create_sqlite_engine(settings.effective_database_url)
            try:
                sessions = create_session_factory(db)
                with sessions() as session:
                    row = session.scalar(select(CaptureRow).where(CaptureRow.id == result["capture_id"]))
                    assert row is not None
                    assert (root / "received" / row.stored_path).read_bytes() == encoded.tobytes()
            finally:
                db.dispose()
            invalid = client.post(
                "/captures?camera_id=../x",
                files={"file": ("test.jpg", encoded.tobytes(), "image/jpeg")},
            )
            assert invalid.status_code == 400
            fake = client.post(
                "/captures?camera_id=cam-01",
                files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")},
            )
            assert fake.status_code == 400
