"""Caso de uso principal: imagem recebida -> detecções -> imagem anotada -> estado da prateleira."""
from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from app.application.errors import InvalidImageError, ModelUnavailableError
from app.application.track_rupture import TrackRupture
from app.domain.detection import Detection
from app.domain.ports import ImageCodec, ImageStore, ObjectDetector, ShelfProfiles
from app.domain.rupture import RuptureEvent
from app.domain.shelf import ShelfEvaluation, ShelfProfile

logger = logging.getLogger(__name__)

# TCC 6.1: a entrada é restrita a image/jpeg e image/png. A extensão acompanha o tipo (corrige a limitação 8-f).
ALLOWED_CONTENT_TYPES = {"image/jpeg": ".jpg", "image/png": ".png"}
CAMERA_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class AnalyzeRequest:
    camera_id: str
    filename: str
    content_type: str
    data: bytes


@dataclass(frozen=True)
class ShelfAnalysis:
    profile: ShelfProfile
    evaluation: ShelfEvaluation
    open_events: dict[str, RuptureEvent | None]


@dataclass(frozen=True)
class AnalysisResult:
    camera_id: str
    filename: str
    detections: tuple[Detection, ...]
    people_count: int | None
    shelf: ShelfAnalysis | None


class AnalyzeImage:
    def __init__(
        self,
        *,
        products: ObjectDetector,
        people: ObjectDetector | None,
        codec: ImageCodec,
        store: ImageStore,
        profiles: ShelfProfiles,
        tracker: TrackRupture,
        clock: Callable[[], datetime],
        max_bytes: int,
    ) -> None:
        self._products = products
        self._people = people
        self._codec = codec
        self._store = store
        self._profiles = profiles
        self._tracker = tracker
        self._clock = clock
        self._max_bytes = max_bytes

    def execute(self, request: AnalyzeRequest) -> AnalysisResult:
        self._validate(request)
        image = self._decode(request.data)
        self._ensure_model_ready()
        detections = self._products.detect(image)
        people_count = self._count_people(image)
        captured_at = self._clock()
        self._persist_images(request, image, detections, captured_at)
        # A regra legada de 60% não identifica produtos por aparência.
        # A ruptura por histórico é calculada pela API usando VisualHistory.
        shelf = None
        return AnalysisResult(
            camera_id=request.camera_id,
            filename=request.filename,
            detections=tuple(detections),
            people_count=people_count,
            shelf=shelf,
        )

    def _validate(self, request: AnalyzeRequest) -> None:
        if not CAMERA_ID_PATTERN.match(request.camera_id):
            raise InvalidImageError("camera_id inválido: use de 1 a 64 caracteres entre letras, números, '_' e '-'")
        if request.content_type not in ALLOWED_CONTENT_TYPES:
            raise InvalidImageError("Tipo de arquivo não suportado: envie image/jpeg ou image/png")
        if not request.data:
            raise InvalidImageError("Arquivo vazio")
        if len(request.data) > self._max_bytes:
            raise InvalidImageError(f"Arquivo maior que o limite de {self._max_bytes} bytes")

    def _decode(self, data: bytes) -> np.ndarray:
        image = self._codec.decode(data)
        if image is None:
            raise InvalidImageError("Não foi possível decodificar a imagem")
        return image

    def _ensure_model_ready(self) -> None:
        if not self._products.is_available:
            raise ModelUnavailableError(
                "Modelo de detecção indisponível: verifique SDP_MODEL_PATH e se o arquivo de pesos existe"
            )

    def _count_people(self, image: np.ndarray) -> int | None:
        if self._people is None or not self._people.is_available:
            return None
        try:
            return len(self._people.detect(image))
        except Exception:  # a falha na contagem de pessoas não deve impedir a análise dos produtos
            logger.exception("Falha ao contar pessoas; fluxo de clientes não registrado nesta captura")
            return None

    def _persist_images(
        self, request: AnalyzeRequest, image: np.ndarray, detections: list[Detection], at: datetime
    ) -> None:
        extension = ALLOWED_CONTENT_TYPES[request.content_type]
        self._store.save_received(request.data, request.camera_id, extension, at)
        self._store.save_annotated(self._codec.annotate(image, detections), request.camera_id, at)

    def _analyze_shelf(
        self, camera_id: str, detections: list[Detection], at: datetime, people: int | None
    ) -> ShelfAnalysis | None:
        profile = self._profiles.get(camera_id)
        if profile is None:
            return None
        counts = Counter(d.class_name for d in detections)
        evaluation = profile.evaluate(counts, self._products.class_names)
        if evaluation.unmonitored:
            logger.warning(
                "Câmera %s: produtos sem classe no modelo carregado (não avaliados): %s",
                camera_id,
                ", ".join(evaluation.unmonitored),
            )
        open_events = self._tracker.execute(profile, evaluation, at, people)
        return ShelfAnalysis(profile=profile, evaluation=evaluation, open_events=open_events)
