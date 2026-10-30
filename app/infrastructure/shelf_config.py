"""Carrega os perfis de gôndola por câmera a partir de um YAML.

Formato:

    cameras:
      cam-01:
        aisle: C10
        products:
          - {class_name: produto_1, category: Mercearia, capacity: 10}
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from app.domain.shelf import ProductSlot, ShelfProfile

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Arquivo de configuração inválido."""


class YamlShelfProfiles:
    def __init__(self, profiles: Mapping[str, ShelfProfile]) -> None:
        self._profiles = dict(profiles)

    @property
    def camera_ids(self) -> list[str]:
        return sorted(self._profiles)

    def get(self, camera_id: str) -> ShelfProfile | None:
        return self._profiles.get(camera_id)

    @classmethod
    def from_file(cls, path: Path, min_occupancy: float) -> YamlShelfProfiles:
        if not path.is_file():
            logger.warning("Arquivo de gôndolas não encontrado (%s): nenhuma regra de ruptura será aplicada", path)
            return cls({})
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return cls(cls._parse(document, min_occupancy))
        except (yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Configuração de gôndolas inválida em {path}: {exc!r}") from exc

    @classmethod
    def _parse(cls, document: Mapping[str, Any], min_occupancy: float) -> dict[str, ShelfProfile]:
        cameras = document.get("cameras") or {}
        return {
            str(camera_id): cls._profile(str(camera_id), entry, min_occupancy)
            for camera_id, entry in cameras.items()
        }

    @staticmethod
    def _profile(camera_id: str, entry: Mapping[str, Any], min_occupancy: float) -> ShelfProfile:
        slots = tuple(
            ProductSlot(
                class_name=str(product["class_name"]),
                category=str(product["category"]),
                capacity=int(product["capacity"]),
            )
            for product in entry["products"]
        )
        return ShelfProfile(
            camera_id=camera_id, aisle=str(entry["aisle"]), slots=slots, min_occupancy=min_occupancy
        )
