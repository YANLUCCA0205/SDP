"""Decodificação de bytes e anotação de imagens com OpenCV/NumPy (TCC, seção 3.5)."""
from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from app.domain.detection import Detection

PALETTE_BGR = (
    (86, 180, 233), (0, 158, 115), (240, 228, 66), (213, 94, 0),
    (204, 121, 167), (0, 114, 178), (230, 159, 0), (140, 140, 140),
)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
TEXT_COLOR_BGR = (0, 0, 0)
LABEL_PADDING = 3


class OpenCvCodec:
    def decode(self, data: bytes) -> np.ndarray | None:
        """Decodifica em modo colorido, garantindo imagem de três canais (TCC 6.1). None se inválida."""
        buffer = np.frombuffer(data, dtype=np.uint8)
        if buffer.size == 0:
            return None
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    def annotate(self, image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
        annotated = image.copy()
        for detection in detections:
            self._draw(annotated, detection)
        return annotated

    def _draw(self, image: np.ndarray, detection: Detection) -> None:
        color = PALETTE_BGR[detection.class_id % len(PALETTE_BGR)]
        box = detection.bbox
        top_left = (int(box.x1), int(box.y1))
        cv2.rectangle(image, top_left, (int(box.x2), int(box.y2)), color, BOX_THICKNESS)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        self._draw_label(image, label, top_left, color)

    @staticmethod
    def _draw_label(image: np.ndarray, label: str, origin: tuple[int, int], color) -> None:
        (width, height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        x, y = origin
        top = max(y - height - baseline - LABEL_PADDING, 0)
        cv2.rectangle(image, (x, top), (x + width + LABEL_PADDING, top + height + baseline + LABEL_PADDING), color, -1)
        cv2.putText(image, label, (x + 1, top + height + 1), FONT, FONT_SCALE, TEXT_COLOR_BGR, 1, cv2.LINE_AA)
