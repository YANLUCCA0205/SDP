"""Armazenamento de imagens em disco (TCC, seções 4.4 e 6.3).

Diferente da versão do TCC (nome fixo, sobrescrito a cada requisição), cada captura recebe nome único com
timestamp, o que mantém histórico e elimina a condição de corrida (limitação 8-c). A "última imagem" continua
disponível por câmera, gravada de forma atômica.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"
LATEST_RECEIVED_STEM = "ultima_imagem"
LATEST_ANNOTATED_NAME = "ultima_deteccao.jpg"
ANNOTATED_EXTENSION = ".jpg"


class FileImageStore:
    def __init__(self, received_dir: Path, results_dir: Path) -> None:
        self._received_dir = received_dir
        self._results_dir = results_dir

    def save_received(self, data: bytes, camera_id: str, extension: str, at: datetime) -> Path:
        directory = self._camera_dir(self._received_dir, camera_id)
        path = directory / f"{at.strftime(TIMESTAMP_FORMAT)}{extension}"
        path.write_bytes(data)
        self._replace_atomically(directory / f"{LATEST_RECEIVED_STEM}{extension}", data=data)
        return path

    def save_annotated(self, image: np.ndarray, camera_id: str, at: datetime) -> Path:
        directory = self._camera_dir(self._results_dir, camera_id)
        path = directory / f"{at.strftime(TIMESTAMP_FORMAT)}{ANNOTATED_EXTENSION}"
        self._write_image(path, image)
        self._replace_atomically(directory / LATEST_ANNOTATED_NAME, image=image)
        return path

    @staticmethod
    def _camera_dir(root: Path, camera_id: str) -> Path:
        directory = root / camera_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Não foi possível gravar a imagem em {path}")

    def _replace_atomically(
        self, target: Path, *, data: bytes | None = None, image: np.ndarray | None = None
    ) -> None:
        temporary = target.with_name(f"{target.stem}.{os.getpid()}.tmp{target.suffix}")
        if data is not None:
            temporary.write_bytes(data)
        else:
            self._write_image(temporary, image)
        os.replace(temporary, target)
