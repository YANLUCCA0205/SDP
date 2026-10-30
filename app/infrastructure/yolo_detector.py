"""Adaptador do Ultralytics YOLO11 (TCC, seções 3.4 e 6.2).

Carrega os pesos uma única vez e adapta os nomes das classes ao modelo (`model.names`),
de modo que trocar yolo11n.pt por best.pt exige apenas alterar a configuração.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from app.domain.detection import BoundingBox, Detection

logger = logging.getLogger(__name__)

CONFIDENCE_DIGITS = 4
COORDINATE_DIGITS = 2


class YoloDetector:
    def __init__(
        self,
        weights_path: Path,
        *,
        confidence: float,
        imgsz: int,
        device: str | None = None,
        class_ids: Sequence[int] | None = None,
    ) -> None:
        self._weights_path = weights_path
        self._confidence = confidence
        self._imgsz = imgsz
        self._device = device
        self._class_ids = list(class_ids) if class_ids is not None else None
        self._model = None
        self._load_error: str | None = None
        # A predição no mesmo objeto de modelo não é garantidamente thread-safe.
        self._lock = threading.Lock()

    @property
    def weights_path(self) -> Path:
        return self._weights_path

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def class_names(self) -> frozenset[str]:
        if self._model is None:
            return frozenset()
        return frozenset(self._names_by_id().values())

    def load(self) -> None:
        """Carrega os pesos. Se ausentes ou inválidos, o detector permanece indisponível (sem fallback)."""
        if not self._weights_path.is_file():
            self._load_error = f"Arquivo de pesos não encontrado: {self._weights_path}"
            logger.warning(self._load_error)
            return
        try:
            from ultralytics import YOLO  # import tardio: a API sobe e informa o problema se a lib faltar

            self._model = YOLO(str(self._weights_path))
            self._load_error = None
            logger.info("Modelo carregado: %s", self._weights_path)
        except Exception as exc:
            self._model = None
            self._load_error = f"Falha ao carregar {self._weights_path}: {exc}"
            logger.exception("Falha ao carregar o modelo %s", self._weights_path)

    def detect(self, image: np.ndarray) -> list[Detection]:
        if self._model is None:
            raise RuntimeError("Modelo não carregado")
        with self._lock:
            results = self._model.predict(image, **self._predict_options())
        return self._to_detections(results[0])

    def _predict_options(self) -> dict:
        options: dict = {"conf": self._confidence, "imgsz": self._imgsz, "verbose": False}
        if self._device:
            options["device"] = self._device
        if self._class_ids is not None:
            options["classes"] = self._class_ids
        return options

    def _names_by_id(self) -> dict[int, str]:
        names = self._model.names
        if self._class_ids is None:
            return dict(names)
        return {class_id: names[class_id] for class_id in self._class_ids if class_id in names}

    def _to_detections(self, result) -> list[Detection]:
        names = self._model.names
        boxes = result.boxes
        coordinates = boxes.xyxy.tolist()
        confidences = boxes.conf.tolist()
        class_ids = boxes.cls.tolist()
        return [
            Detection(
                class_id=int(class_id),
                class_name=names[int(class_id)],
                confidence=round(float(confidence), CONFIDENCE_DIGITS),
                bbox=BoundingBox(*(round(float(c), COORDINATE_DIGITS) for c in xyxy)),
            )
            for xyxy, confidence, class_id in zip(coordinates, confidences, class_ids)
        ]
