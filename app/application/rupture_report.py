"""Consulta de eventos de ruptura no formato da base descrita no TCC (seção 2.5-b)."""
from __future__ import annotations

from typing import Any

from app.domain.flow import FlowThresholds
from app.domain.ports import EventRepository
from app.domain.rupture import RuptureEvent

DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M:%S"
STATUS_PENDING = "pendente"
STATUS_OPEN = "confirmada_em_andamento"
STATUS_CLOSED = "encerrada"


class RuptureReport:
    def __init__(self, events: EventRepository, thresholds: FlowThresholds) -> None:
        self._events = events
        self._thresholds = thresholds

    def records(
        self,
        *,
        camera_id: str | None = None,
        confirmed_only: bool = True,
        open_only: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        events = self._events.list(
            camera_id=camera_id, confirmed_only=confirmed_only, open_only=open_only, limit=limit
        )
        return [self.to_record(event) for event in events]

    def to_record(self, event: RuptureEvent) -> dict[str, Any]:
        return {
            "id_ruptura": event.id,
            "camera_id": event.camera_id,
            "data": event.started_at.strftime(DATE_FORMAT),
            "h_inicio": event.started_at.strftime(TIME_FORMAT),
            "h_fim": event.ended_at.strftime(TIME_FORMAT) if event.ended_at else None,
            "tempo_ruptura_min": event.duration_minutes,
            "produto": event.product,
            "categoria": event.category,
            "corredor": event.aisle,
            "funcionario": event.employee,
            "fluxo_de_cliente": self._flow_label(event),
            "status": self._status(event),
        }

    def _flow_label(self, event: RuptureEvent) -> str | None:
        average = event.average_people
        return None if average is None else self._thresholds.classify(average).value

    @staticmethod
    def _status(event: RuptureEvent) -> str:
        if not event.confirmed:
            return STATUS_PENDING
        return STATUS_OPEN if event.is_open else STATUS_CLOSED
