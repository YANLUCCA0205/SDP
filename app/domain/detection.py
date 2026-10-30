"""Value objects produzidos pelo detector de objetos (TCC, seção 4.3)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Caixa delimitadora em pixels: (x1, y1) canto superior esquerdo, (x2, y2) inferior direito."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Detection:
    """Uma detecção: índice e nome da classe, confiança de 0 a 1 e caixa."""

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
