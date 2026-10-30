"""Transforma leituras de prateleira em eventos de ruptura persistidos."""
from __future__ import annotations

import threading
from datetime import datetime

from app.domain.ports import EventRepository, ShiftRepository
from app.domain.rupture import RuptureEvent
from app.domain.shelf import ProductReading, ShelfEvaluation, ShelfProfile

EMPLOYEE_SEPARATOR = " / "


class TrackRupture:
    """Mantém o ciclo de vida dos eventos. Serializa o processamento por câmera.

    O lock é por processo: rode a API com um único worker (padrão) ou mova a serialização para o banco.
    """

    def __init__(
        self, events: EventRepository, shifts: ShiftRepository, confirmations_required: int
    ) -> None:
        if confirmations_required < 1:
            raise ValueError("confirmations_required deve ser >= 1")
        self._events = events
        self._shifts = shifts
        self._confirmations_required = confirmations_required
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def execute(
        self,
        profile: ShelfProfile,
        evaluation: ShelfEvaluation,
        at: datetime,
        people: int | None,
    ) -> dict[str, RuptureEvent | None]:
        """Retorna, por produto, o evento aberto (pendente ou confirmado) após esta captura, ou None."""
        with self._lock_for(profile.camera_id):
            return {
                reading.product: self._apply(profile, reading, at, people)
                for reading in evaluation.readings
            }

    def _lock_for(self, camera_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(camera_id, threading.Lock())

    def _apply(
        self, profile: ShelfProfile, reading: ProductReading, at: datetime, people: int | None
    ) -> RuptureEvent | None:
        open_event = self._events.find_open(profile.camera_id, reading.product)
        if reading.in_rupture:
            return self._on_rupture(profile, reading, open_event, at, people)
        self._on_available(profile, open_event, at)
        return None

    def _on_rupture(
        self,
        profile: ShelfProfile,
        reading: ProductReading,
        open_event: RuptureEvent | None,
        at: datetime,
        people: int | None,
    ) -> RuptureEvent:
        if open_event is None:
            return self._events.add(
                RuptureEvent.open(
                    camera_id=profile.camera_id,
                    aisle=profile.aisle,
                    product=reading.product,
                    category=reading.slot.category,
                    at=at,
                    people=people,
                    confirmations_required=self._confirmations_required,
                )
            )
        open_event.register_capture(people, self._confirmations_required)
        self._events.update(open_event)
        return open_event

    def _on_available(
        self, profile: ShelfProfile, open_event: RuptureEvent | None, at: datetime
    ) -> None:
        if open_event is None:
            return
        if not open_event.confirmed:
            self._events.delete(open_event.id)  # falso positivo: não chegou a ser confirmado
            return
        open_event.close(at, self._responsible_at(profile.aisle, at))
        self._events.update(open_event)

    def _responsible_at(self, aisle: str, at: datetime) -> str | None:
        employees = self._shifts.employees_on_duty(aisle, at)
        return EMPLOYEE_SEPARATOR.join(employees) if employees else None
