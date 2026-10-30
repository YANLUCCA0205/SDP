"""Gôndola, produtos esperados e leitura do estado (TCC, seções 2.7 e 4.5).

O YOLO apenas detecta. Este módulo decide, a partir das contagens, se cada produto
está disponível ou em possível ruptura. Regra de negócio informada pelo autor: a gôndola
deve estar com ao menos `min_occupancy` (60%) da capacidade de cada produto.
"""
from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import Enum

PRECISION_DIGITS = 9


class ShelfState(str, Enum):
    DISPONIVEL = "disponivel"
    POSSIVEL_RUPTURA = "possivel_ruptura"


@dataclass(frozen=True)
class ProductSlot:
    """Produto monitorado em uma gôndola. `capacity` é o máximo de unidades visíveis."""

    class_name: str
    category: str
    capacity: int

    def __post_init__(self) -> None:
        if not self.class_name.strip():
            raise ValueError("class_name não pode ser vazio")
        if self.capacity <= 0:
            raise ValueError(f"capacity de '{self.class_name}' deve ser maior que zero")

    def required_units(self, min_occupancy: float) -> int:
        # round() evita que erros de ponto flutuante (ex.: 6.000000000000001) elevem o ceil.
        return math.ceil(round(self.capacity * min_occupancy, PRECISION_DIGITS))


@dataclass(frozen=True)
class ProductReading:
    slot: ProductSlot
    detected: int
    required: int

    @property
    def product(self) -> str:
        return self.slot.class_name

    @property
    def in_rupture(self) -> bool:
        return self.detected < self.required

    @property
    def state(self) -> ShelfState:
        return ShelfState.POSSIVEL_RUPTURA if self.in_rupture else ShelfState.DISPONIVEL


@dataclass(frozen=True)
class ShelfEvaluation:
    readings: tuple[ProductReading, ...]
    unmonitored: tuple[str, ...]


@dataclass(frozen=True)
class ShelfProfile:
    """Configuração de uma câmera: corredor e produtos esperados na gôndola apontada."""

    camera_id: str
    aisle: str
    slots: tuple[ProductSlot, ...]
    min_occupancy: float

    def __post_init__(self) -> None:
        if not 0 < self.min_occupancy <= 1:
            raise ValueError("min_occupancy deve estar no intervalo (0, 1]")
        if not self.slots:
            raise ValueError(f"A câmera '{self.camera_id}' não possui produtos configurados")

    def evaluate(
        self, counts: Mapping[str, int], known_classes: Collection[str]
    ) -> ShelfEvaluation:
        """Compara contagens com o mínimo esperado.

        Produtos que o modelo carregado não conhece não são avaliados: contá-los como zero
        geraria uma ruptura falsa (ex.: modelo genérico yolo11n.pt com perfil de produtos reais).
        """
        monitored = [s for s in self.slots if s.class_name in known_classes]
        unmonitored = tuple(s.class_name for s in self.slots if s.class_name not in known_classes)
        readings = tuple(self._read(slot, counts) for slot in monitored)
        return ShelfEvaluation(readings=readings, unmonitored=unmonitored)

    def _read(self, slot: ProductSlot, counts: Mapping[str, int]) -> ProductReading:
        return ProductReading(
            slot=slot,
            detected=counts.get(slot.class_name, 0),
            required=slot.required_units(self.min_occupancy),
        )
