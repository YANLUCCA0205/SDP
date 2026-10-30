"""Interfaces (portas) entre o domínio/aplicação e a infraestrutura."""
from __future__ import annotations

from collections.abc import Collection, Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from app.domain.detection import Detection
from app.domain.rupture import RuptureEvent
from app.domain.shelf import ShelfProfile
from app.domain.shift import Shift


class ObjectDetector(Protocol):
    @property
    def is_available(self) -> bool: ...

    @property
    def class_names(self) -> Collection[str]: ...

    def detect(self, image: np.ndarray) -> list[Detection]: ...


class ImageCodec(Protocol):
    def decode(self, data: bytes) -> np.ndarray | None: ...

    def annotate(self, image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray: ...


class ImageStore(Protocol):
    def save_received(self, data: bytes, camera_id: str, extension: str, at: datetime) -> Path: ...

    def save_annotated(self, image: np.ndarray, camera_id: str, at: datetime) -> Path: ...


class ShelfProfiles(Protocol):
    def get(self, camera_id: str) -> ShelfProfile | None: ...


class EventRepository(Protocol):
    def add(self, event: RuptureEvent) -> RuptureEvent: ...

    def update(self, event: RuptureEvent) -> None: ...

    def delete(self, event_id: int) -> None: ...

    def find_open(self, camera_id: str, product: str) -> RuptureEvent | None: ...

    def list(
        self,
        *,
        camera_id: str | None = None,
        confirmed_only: bool = True,
        open_only: bool = False,
        limit: int = 100,
    ) -> list[RuptureEvent]: ...


class ShiftRepository(Protocol):
    def add(self, shift: Shift) -> Shift: ...

    def list(self) -> list[Shift]: ...

    def delete(self, shift_id: int) -> bool: ...

    def employees_on_duty(self, aisle: str, at: datetime) -> list[str]: ...
