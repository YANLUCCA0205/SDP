"""Baixa os pesos pré-treinados yolo11n.pt (COCO) para models/, usando a API oficial do Ultralytics.

Requer acesso à internet na máquina onde for executado. Uso: python scripts/download_weights.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
WEIGHTS_NAME = "yolo11n.pt"


def main() -> None:
    target = MODELS_DIR / WEIGHTS_NAME
    if target.is_file():
        print(f"{target} já existe.")
        return
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    YOLO(WEIGHTS_NAME)  # dispara o download automático
    candidates = [Path.cwd() / WEIGHTS_NAME, Path(SETTINGS["weights_dir"]) / WEIGHTS_NAME]
    downloaded = next((p for p in candidates if p.is_file()), None)
    if downloaded is None:
        sys.exit(f"Download concluído, mas {WEIGHTS_NAME} não foi localizado em: {[str(p) for p in candidates]}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, target)
    print(f"Pesos copiados para {target}")


if __name__ == "__main__":
    main()
