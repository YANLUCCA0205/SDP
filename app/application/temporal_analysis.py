"""Analise temporal por prateleira para capturas historicas do SDP.

Combina mascaras de mudanca de ocupacao com as deteccoes do YoloDetector
existente. Deteccoes COCO nao sao tratadas como identificacao de SKU.
A confirmacao e uma persistencia visual, nao validacao comercial de ruptura.
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import yaml

from app.infrastructure.capture_store import CAMERA_PATTERN
from app.infrastructure.opencv_codec import OpenCvCodec

DEFAULT_CONFIG = Path("config/temporal_shelves.yaml")


def _get_bands(path: Path, camera_id: str, shape: tuple[int, ...]) -> dict[str, tuple[int, int, int, int]]:
    if not path.is_file():
        raise ValueError(f"Configuracao de prateleiras ausente: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cameras = data.get("cameras", {})
    bands = cameras.get(camera_id)
    if not isinstance(bands, dict) or not bands:
        raise ValueError(f"Camera sem regioes configuradas: {camera_id}")
    height, width = shape[:2]
    output = {}
    for label, rect in bands.items():
        if not isinstance(rect, list) or len(rect) != 4 or any(not isinstance(v, (int, float)) for v in rect):
            raise ValueError(f"Regiao invalida: {label}")
        x1, y1, x2, y2 = (int(rect[0] * width), int(rect[1] * height),
                          int(rect[2] * width), int(rect[3] * height))
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Coordenadas invalidas: {label}")
        output[str(label)] = (x1, y1, x2, y2)
    return output


def _mask(reference_lab: np.ndarray, current_lab: np.ndarray, rect: tuple[int, int, int, int]):
    x1, y1, x2, y2 = rect
    base = reference_lab[y1:y2, x1:x2].astype(np.int16)
    now = current_lab[y1:y2, x1:x2].astype(np.int16)
    # Normaliza somente a componente de luminosidade, protegendo contra mudancas globais.
    offset = int(np.clip(np.median(base - now), -20, 20))
    adjusted = np.clip(now + offset, 0, 255)
    change = (base >= 98) & (adjusted <= 92) & (base - adjusted >= 34)
    binary = np.uint8(change) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    boxes = []
    for index in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[index]]
        if area >= 110 and w >= 10 and h >= 12:
            boxes.append({"x1": x + x1, "y1": y + y1,
                          "x2": x + x1 + w, "y2": y + y1 + h, "area_px": area})
    return binary, boxes


def analyze_history(*, camera_id: str, day: str, reference_prefix: str,
                    received_root: Path, results_root: Path, config_path: Path,
                    detector=None) -> dict:
    if not CAMERA_PATTERN.fullmatch(camera_id):
        raise ValueError("camera_id invalido")
    try:
        parsed_day = date.fromisoformat(day)
    except ValueError as exc:
        raise ValueError("data invalida: use AAAA-MM-DD") from exc
    if parsed_day.isoformat() != day:
        raise ValueError("data invalida: use AAAA-MM-DD")
    if not (reference_prefix.isdigit() and len(reference_prefix) == 6):
        raise ValueError("reference_prefix deve ser HHMMSS (6 digitos)")
    source = received_root / camera_id / day
    photos = sorted(p for p in source.glob("*.jpg") if p.name[:6].isdigit())
    reference_file = next((p for p in photos if p.name.startswith(reference_prefix + "_")), None)
    if reference_file is None:
        raise ValueError("Captura de referencia nao encontrada")
    photos = [p for p in photos if p.name >= reference_file.name]
    reference = cv2.imread(str(reference_file))
    if reference is None:
        raise ValueError("Nao foi possivel ler a imagem de referencia")
    bands = _get_bands(config_path, camera_id, reference.shape)
    output = results_root / "temporal" / camera_id / day
    output.mkdir(parents=True, exist_ok=True)
    codec = OpenCvCodec()
    reference_lab = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)[:, :, 0), (5, 5), 0)
    counts = {name: 0 for name in bands}
    rows = []
    for photo in photos:
        image = cv2.imread(str(photo))
        if image is None or image.shape != reference.shape:
            continue
        now_lab = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0), (5, 5), 0)
        # Reutiliza a interface do detector do SDP quando os pesos ja estao carregados.
        detections = detector.detect(image) if detector is not None and detector.is_available else []
        annotated = codec.annotate(image, detections)
        full_mask = np.zeros(reference.shape[:2], np.uint8)
        shelves = []
        for name, rect in bands.items():
            mask, boxes = _mask(reference_lab, now_lab, rect)
            x1, y1, x2, y2 = rect
            full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], mask)
            counts[name] = counts[name] + 1 if boxes else 0
            status = ("PERSISTENTE" if counts[name] >= 2 else
                      "SUSPEITA" if boxes else "SEM_ALERTA")
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 170, 0), 1)
            cv2.putText(annotated, name, (x1 + 2, y1 + 13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (255, 255, 0), 1)
            for box in boxes:
                cv2.rectangle(annotated, (box["x1"], box["y1"]),
                              (box["x2"], box["y2"]), (0, 0, 255), 2)
            shelves.append({"shelf": name, "status": status, "regions": boxes})
            rows.append({"capture": photo.name, "shelf": name,
                         "status": status, "regions": len(boxes)})
        cv2.imwrite(str(output / f"{photo.stem}_analise.jpg"), annotated)
        cv2.imwrite(str(output / f"{photo.stem}_mascara.png"), full_mask)
        # Nao transforma objeto COCO em produto do supermercado.
        rows[-1]["yolo_detections"] = len(detections)
    csv_path = output / "analise.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["capture", "shelf", "status", "regions", "yolo_detections"],
                                delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return {"status": "analyzed", "camera_id": camera_id, "date": day,
            "reference": reference_file.name, "processed_captures": len(rows) // len(bands),
            "model_loaded": bool(detector is not None and detector.is_available),
            "results_dir": str(output), "csv": str(csv_path),
            "warning": "PERSISTENTE indica alteracao visual em 2 capturas; nao comprova ruptura nem SKU.",
            "readings": rows}
