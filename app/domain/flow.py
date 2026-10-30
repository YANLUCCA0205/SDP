"""Fluxo de clientes: pessoas dentro do raio de detecção da câmera (definição do autor)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CustomerFlow(str, Enum):
    BAIXO = "baixo"
    MEDIO = "medio"
    ALTO = "alto"


@dataclass(frozen=True)
class FlowThresholds:
    """Limites sobre a média de pessoas por captura: até `low_max` é baixo, até `medium_max` é médio."""

    low_max: float
    medium_max: float

    def __post_init__(self) -> None:
        if not 0 <= self.low_max <= self.medium_max:
            raise ValueError("É necessário 0 <= low_max <= medium_max")

    def classify(self, average_people: float) -> CustomerFlow:
        if average_people <= self.low_max:
            return CustomerFlow.BAIXO
        if average_people <= self.medium_max:
            return CustomerFlow.MEDIO
        return CustomerFlow.ALTO
