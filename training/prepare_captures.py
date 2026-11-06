"""Prepara capturas reais para anotacao YOLO, sem inventar rotulos.

Uso: python training/prepare_captures.py --camera-id cam-01 --day 2026-11-03
Gera dataset/to_annotate/<camera>/<data> sem modificar received/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
from datetime import date
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CAMERA_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def prepare(source: Path, target: Path) -> dict[str, int | str]:
    if not source.is_dir():
        raise ValueError(f"Pasta de capturas nao encontrada: {source}")
    if source.resolve() == target.resolve():
        raise ValueError("Origem e destino nao podem ser iguais")
    paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"Nenhuma imagem encontrada: {source}")
    images_dir = target / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    hashes = set()
    duplicates = 0
    invalid = 0
    for path in paths:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        image = cv2.imread(str(path))
        if image is None:
            invalid += 1
            manifest.append({"filename": path.name, "sha256": digest, "width": "", "height": "",
                             "status": "INVALID_IMAGE", "label_status": "NOT_ANNOTATED"})
            continue
        if digest in hashes:
            duplicates += 1
            status = "DUPLICATE"
        else:
            status = "READY_FOR_ANNOTATION"
            hashes.add(digest)
            shutil.copy2(path, images_dir / path.name)
        height, width = image.shape[:2]
        manifest.append({"filename": path.name, "sha256": digest, "width": width, "height": height,
                         "status": status, "label_status": "NOT_ANNOTATED"})
    with (target / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["filename", "sha256", "width", "height", "status", "label_status"], delimiter=";")
        writer.writeheader()
        writer.writerows(manifest)
    return {"found": len(paths), "copied": len(hashes), "duplicates": duplicates,
            "invalid": invalid, "destination": str(target)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara fotografias da ESP32-CAM para anotacao YOLO")
    parser.add_argument("--camera-id", default="cam-01")
    parser.add_argument("--day", default="2026-11-03")
    parser.add_argument("--received-root", type=Path, default=ROOT / "received")
    parser.add_argument("--output-root", type=Path, default=ROOT / "dataset" / "to_annotate")
    args = parser.parse_args()
    if not CAMERA_PATTERN.fullmatch(args.camera_id):
        parser.error("camera-id invalido")
    try:
        if date.fromisoformat(args.day).isoformat() != args.day:
            raise ValueError
    except ValueError:
        parser.error("day deve estar em AAAA-MM-DD")
    result = prepare(args.received_root / args.camera_id / args.day,
                     args.output_root / args.camera_id / args.day)
    for key, value in result.items():
        print(f"{key}: {value}")
    print("Rotulos nao foram gerados. Anote objetos reais antes de treinar.")


if __name__ == "__main__":
    main()
