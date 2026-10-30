"""Aquisição histórica de imagens, independente de inferência YOLO.

A tabela é registrada no mesmo metadata SQLAlchemy do SDP, antes do create_all.
Uma captura recebida não é automaticamente uma análise de ruptura.
"""
from __future__ import annotations

import hashlib
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker

from app.infrastructure.sql_repositories import Base

CAMERA_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png"}


class CaptureRow(Base):
    __tablename__ = "captures"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    received_at_utc: Mapped[datetime] = mapped_column(DateTime, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(1024))
    content_type: Mapped[str] = mapped_column(String(32))
    bytes_count: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    processing_status: Mapped[str] = mapped_column(String(24), default="PENDING")


class CaptureStore:
    def __init__(self, root: Path, sessions: sessionmaker[Session], max_bytes: int) -> None:
        self.root = root
        self.sessions = sessions
        self.max_bytes = max_bytes

    def receive(self, *, camera_id: str, filename: str, content_type: str, data: bytes) -> dict:
        if not CAMERA_PATTERN.fullmatch(camera_id):
            raise ValueError("camera_id inválido")
        extension = IMAGE_TYPES.get(content_type)
        if extension is None:
            raise ValueError("Envie image/jpeg ou image/png")
        if not data or len(data) > self.max_bytes:
            raise ValueError("Imagem vazia ou acima do limite")
        # Confirma conteúdo real, não somente o cabeçalho informado pelo cliente.
        if content_type == "image/jpeg" and not data.startswith(b"\\xff\\xd8\\xff"):
            raise ValueError("Conteúdo não corresponde a JPEG")
        if content_type == "image/png" and not data.startswith(b"\\x89PNG\\r\\n\\x1a\\n"):
            raise ValueError("Conteúdo não corresponde a PNG")
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Imagem inválida ou corrompida")

        capture_id = uuid.uuid4().hex
        at = datetime.now(timezone.utc).replace(tzinfo=None)
        folder = self.root / camera_id / at.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{at.strftime('%H%M%S_%f')}_{capture_id}{extension}"
        # O nome gerado é exclusivo; arquivo criado sem permitir sobrescrita.
        with path.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        row = CaptureRow(
            id=capture_id, camera_id=camera_id, received_at_utc=at,
            filename=filename[:255], stored_path=str(path.relative_to(self.root)),
            content_type=content_type, bytes_count=len(data),
            sha256=hashlib.sha256(data).hexdigest(), processing_status="PENDING",
        )
        try:
            with self.sessions.begin() as session:
                session.add(row)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return {
            "status": "stored", "capture_id": capture_id,
            "camera_id": camera_id, "received_at_utc": at.isoformat() + "Z",
            "bytes_count": len(data), "processing_status": "PENDING",
        }
