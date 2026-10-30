"""Evento de ruptura e seu ciclo de vida (TCC, seção 2.7).

- Início: primeira captura em estado de ruptura.
- Confirmação: exige mais de uma captura consecutiva em ruptura (configurável) para reduzir falsos positivos.
- Fim: primeira captura que volta a indicar disponibilidade.
- Duração: diferença entre os dois instantes (resolução limitada pelo intervalo de captura).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

SECONDS_PER_MINUTE = 60
DURATION_DIGITS = 2


@dataclass
class RuptureEvent:
    camera_id: str
    aisle: str
    product: str
    category: str
    started_at: datetime
    consecutive_captures: int = 0
    people_total: int = 0
    people_samples: int = 0
    confirmed: bool = False
    ended_at: datetime | None = None
    employee: str | None = None
    id: int | None = None

    @classmethod
    def open(
        cls,
        *,
        camera_id: str,
        aisle: str,
        product: str,
        category: str,
        at: datetime,
        people: int | None,
        confirmations_required: int,
    ) -> RuptureEvent:
        event = cls(camera_id=camera_id, aisle=aisle, product=product, category=category, started_at=at)
        event.register_capture(people, confirmations_required)
        return event

    @property
    def is_open(self) -> bool:
        return self.ended_at is None

    def register_capture(self, people: int | None, confirmations_required: int) -> None:
        self.consecutive_captures += 1
        self._register_people(people)
        if self.consecutive_captures >= confirmations_required:
            self.confirmed = True

    def close(self, at: datetime, employee: str | None) -> None:
        if not self.confirmed:
            raise ValueError("Somente rupturas confirmadas podem ser encerradas; as pendentes são descartadas")
        if at < self.started_at:
            raise ValueError("O fim da ruptura não pode ser anterior ao início")
        self.ended_at = at
        self.employee = employee

    @property
    def duration_minutes(self) -> float | None:
        if self.ended_at is None:
            return None
        seconds = (self.ended_at - self.started_at).total_seconds()
        return round(seconds / SECONDS_PER_MINUTE, DURATION_DIGITS)

    @property
    def average_people(self) -> float | None:
        """Média de pessoas por captura durante o evento; None se a contagem nunca esteve disponível."""
        if self.people_samples == 0:
            return None
        return self.people_total / self.people_samples

    def _register_people(self, people: int | None) -> None:
        if people is None:
            return
        self.people_total += people
        self.people_samples += 1
