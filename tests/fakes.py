"""Dublês de teste: substituem apenas o detector e a persistência em memória, isolando regras de negócio."""
from __future__ import annotations

import copy
from collections.abc import Iterable
from datetime import datetime

import numpy as np

from app.domain.detection import BoundingBox, Detection
from app.domain.rupture import RuptureEvent
from app.domain.shelf import ProductSlot, ShelfProfile
from app.domain.shift import Shift

T0 = datetime(2026, 6, 10, 8, 0, 0)


def detection(class_name: str, class_id: int = 0, confidence: float = 0.9) -> Detection:
    return Detection(class_id, class_name, confidence, BoundingBox(10, 10, 60, 60))


class FakeDetector:
    def __init__(
        self,
        detections: Iterable[Detection] = (),
        class_names: Iterable[str] = (),
        available: bool = True,
        fail: bool = False,
    ) -> None:
        self.detections = list(detections)
        self._class_names = frozenset(class_names)
        self.is_available = available
        self.fail = fail
        self.calls = 0

    @property
    def class_names(self) -> frozenset[str]:
        return self._class_names

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("falha simulada do detector")
        return list(self.detections)


class InMemoryEventRepository:
    def __init__(self) -> None:
        self._events: dict[int, RuptureEvent] = {}
        self._next_id = 1

    def add(self, event: RuptureEvent) -> RuptureEvent:
        event.id = self._next_id
        self._next_id += 1
        self._events[event.id] = copy.deepcopy(event)
        return event

    def update(self, event: RuptureEvent) -> None:
        self._events[event.id] = copy.deepcopy(event)

    def delete(self, event_id: int) -> None:
        self._events.pop(event_id, None)

    def find_open(self, camera_id: str, product: str) -> RuptureEvent | None:
        for event in self._events.values():
            if event.camera_id == camera_id and event.product == product and event.is_open:
                return copy.deepcopy(event)
        return None

    def list(self, *, camera_id=None, confirmed_only=True, open_only=False, limit=100) -> list[RuptureEvent]:
        events = [
            copy.deepcopy(e)
            for e in self._events.values()
            if (camera_id is None or e.camera_id == camera_id)
            and (not confirmed_only or e.confirmed)
            and (not open_only or e.is_open)
        ]
        return sorted(events, key=lambda e: e.started_at, reverse=True)[:limit]


class InMemoryShiftRepository:
    def __init__(self, shifts: Iterable[Shift] = ()) -> None:
        self._shifts = list(shifts)

    def add(self, shift: Shift) -> Shift:
        shift.id = len(self._shifts) + 1
        self._shifts.append(shift)
        return shift

    def list(self) -> list[Shift]:
        return list(self._shifts)

    def delete(self, shift_id: int) -> bool:
        before = len(self._shifts)
        self._shifts = [s for s in self._shifts if s.id != shift_id]
        return len(self._shifts) < before

    def employees_on_duty(self, aisle: str, at: datetime) -> list[str]:
        return [s.employee for s in self._shifts if s.aisle == aisle and s.covers(at)]


class StaticProfiles:
    def __init__(self, *profiles: ShelfProfile) -> None:
        self._profiles = {p.camera_id: p for p in profiles}

    def get(self, camera_id: str) -> ShelfProfile | None:
        return self._profiles.get(camera_id)


def cafe_profile(camera_id: str = "cam-01", capacity: int = 10) -> ShelfProfile:
    """Perfil de teste com um produto; mínimo = ceil(capacity * 0.6)."""
    return ShelfProfile(
        camera_id=camera_id,
        aisle="C10",
        slots=(ProductSlot("cafe", "Mercearia", capacity),),
        min_occupancy=0.6,
    )
