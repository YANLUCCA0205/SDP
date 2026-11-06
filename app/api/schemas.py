"""Modelos de entrada e saída da API (contrato do TCC, seção 4.3, com campos aditivos opcionais)."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from app.application.analyze_image import AnalysisResult
from app.domain.detection import Detection


class BoundingBoxOut(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionOut(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBoxOut

    @classmethod
    def from_domain(cls, detection: Detection) -> DetectionOut:
        return cls(
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=round(detection.confidence, 4),
            bbox=BoundingBoxOut(**vars(detection.bbox)),
        )


class ProductReadingOut(BaseModel):
    class_name: str
    category: str
    detected: int
    required: int
    state: str
    rupture_id: int | None = None
    rupture_confirmed: bool | None = None


class ShelfOut(BaseModel):
    aisle: str
    min_occupancy: float
    products: list[ProductReadingOut]
    unmonitored_classes: list[str]


class AnalyzeResponse(BaseModel):
    status: str
    filename: str
    detections_count: int
    detections: list[DetectionOut]
    camera_id: str
    people_count: int | None = None
    shelf: ShelfOut | None = None

    @classmethod
    def from_result(cls, result: AnalysisResult) -> AnalyzeResponse:
        return cls(
            status="ok",
            filename=result.filename,
            detections_count=len(result.detections),
            detections=[DetectionOut.from_domain(d) for d in result.detections],
            camera_id=result.camera_id,
            people_count=result.people_count,
            shelf=_shelf_out(result),
        )


def _shelf_out(result: AnalysisResult) -> ShelfOut | None:
    shelf = result.shelf
    if shelf is None:
        return None
    products = []
    for reading in shelf.evaluation.readings:
        event = shelf.open_events.get(reading.product)
        products.append(
            ProductReadingOut(
                class_name=reading.product,
                category=reading.slot.category,
                detected=reading.detected,
                required=reading.required,
                state=reading.state.value,
                rupture_id=event.id if event else None,
                rupture_confirmed=event.confirmed if event else None,
            )
        )
    return ShelfOut(
        aisle=shelf.profile.aisle,
        min_occupancy=shelf.profile.min_occupancy,
        products=products,
        unmonitored_classes=list(shelf.evaluation.unmonitored),
    )


class ShiftIn(BaseModel):
    employee: str = Field(min_length=1, max_length=120)
    aisle: str = Field(min_length=1, max_length=120)
    starts_at: datetime
    ends_at: datetime

    @model_validator(mode="after")
    def _check_interval(self) -> ShiftIn:
        if self.ends_at <= self.starts_at:
            raise ValueError("ends_at deve ser posterior a starts_at")
        return self


class ShiftOut(ShiftIn):
    id: int
