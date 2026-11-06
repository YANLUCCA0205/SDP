"""Teste isolado de mudança visual na gondola da camera cam-01.

Executar na raiz do projeto: python scripts/teste_ruptura_sdp.py
Nao altera API, banco, firmware nem imagens originais.
Os alertas sao candidatos visuais, nao rupturas comprovadas por YOLO.
"""
from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "received" / "cam-01" / "2026-11-03"
OUT = ROOT / "runs" / "rupture_poc" / "cam-01"
REFERENCE_PREFIX = "221525_"
# Regiao fixa: terceira prateleira, apenas para o enquadramento deste teste.
ROI = (175, 258, 565, 366)
MIN_AREA = 450
MIN_WIDTH = 18
MIN_HEIGHT = 20
DARK_LIMIT = 78
BRIGHT_REF = 100
DROP = 42
CONFIRM_FRAMES = 2


def load(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Imagem invalida: {path}")
    return image


def candidates(reference, current):
    x1, y1, x2, y2 = ROI
    base = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)[:, :, 0], (5, 5), 0)
    now = cv2.GaussianBlur(cv2.cvtColor(current, cv2.COLOR_BGR2LAB)[:, :, 0], (5, 5), 0)
    base_roi = base[y1:y2, x1:x2].astype(np.int16)
    now_roi = now[y1:y2, x1:x2].astype(np.int16)
    shift = int(np.clip(np.median(base_roi - now_roi), -20, 20))
    corrected = np.clip(now_roi + shift, 0, 255)
    mask = ((base_roi >= BRIGHT_REF) & (corrected < DARK_LIMIT)
            & ((base_roi - corrected) >= DROP)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if area >= MIN_AREA and w >= MIN_WIDTH and h >= MIN_HEIGHT:
            boxes.append((x + x1, y + y1, w, h, area))
    return boxes, mask


def main():
    files = sorted(SRC.glob("*.jpg"))
    if not files:
        raise SystemExit(f"Nenhuma imagem encontrada: {SRC}")
    ref_file = next((p for p in files if p.name.startswith(REFERENCE_PREFIX)), None)
    if ref_file is None:
        raise SystemExit("Imagem de referencia 22:15:25 nao encontrada")
    reference = load(ref_file)
    OUT.mkdir(parents=True, exist_ok=True)
    consecutive = 0
    rows = []
    for path in (p for p in files if p.name >= ref_file.name):
        image = load(path)
        if image.shape != reference.shape:
            raise SystemExit(f"Resolucao diferente: {path.name}")
        boxes, mask = candidates(reference, image)
        consecutive = consecutive + 1 if boxes else 0
        status = ("CONFIRMADA" if consecutive >= CONFIRM_FRAMES
                  else "SUSPEITA" if boxes else "SEM_ALERTA")
        annotated = image.copy()
        x1, y1, x2, y2 = ROI
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 190, 0), 1)
        for x, y, w, h, area in boxes:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(annotated, f"{area}px", (x, max(18, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
        cv2.rectangle(annotated, (0, 0), (image.shape[1], 28), (20, 20, 20), -1)
        cv2.putText(annotated, f"{path.name[:6]} | {status} | regioes={len(boxes)}",
                    (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (255, 255, 255), 1)
        cv2.imwrite(str(OUT / f"{path.name[:6]}_resultado.jpg"), annotated)
        cv2.imwrite(str(OUT / f"{path.name[:6]}_mascara.png"), mask)
        rows.append({"captura": path.name, "status": status, "regioes": len(boxes),
                     "maior_area_px": max((b[4] for b in boxes), default=0),
                     "caixas": str(boxes)})
        print(f"{path.name[:6]} | {status:<12} | regioes={len(boxes)}")
    with (OUT / "resultado.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Resultados em: {OUT}")
    print("ATENCAO: comparar manualmente. Alertas visuais nao provam ruptura.")


if __name__ == "__main__":
    main()
