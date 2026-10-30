"""Escala de trabalho usada para identificar o responsável pela reposição (TCC, seção 2.4).

A identificação cruza o horário do evento com a escala. Não há reconhecimento facial.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Shift:
    employee: str
    aisle: str
    starts_at: datetime
    ends_at: datetime
    id: int | None = None

    def __post_init__(self) -> None:
        if not self.employee.strip():
            raise ValueError("employee não pode ser vazio")
        if not self.aisle.strip():
            raise ValueError("aisle não pode ser vazio")
        if self.ends_at <= self.starts_at:
            raise ValueError("ends_at deve ser posterior a starts_at")

    def covers(self, moment: datetime) -> bool:
        return self.starts_at <= moment < self.ends_at
