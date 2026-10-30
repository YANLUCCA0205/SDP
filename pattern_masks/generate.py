
from pathlib import Path
import json

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "runs/train/teste_inicial/weights/best.pt"
SOURCE = ROOT / "testes_reais/foto_1.jpg"
OUTPUT = ROOT / "runs/pattern_masks/foto_1"

CONF = 0.35
IMGSZ = 1280
SIMILARITY = 0.90


def describe(crop):
    """Extrai um histograma de cores HSV normalizado."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None, [24, 16],
        [0, 180, 0, 256]
    )
    cv2.normalize(hist, hist)
    return hist.astype(np.float32)


def similarity(a, b):
    score = cv2.compareHist(
        a, b, cv2.HISTCMP_CORREL
    )
    return float(score)


def main():
    if not MODEL.exists():
        raise FileNotFoundError(f"Modelo ausente: {MODEL}")

    image = cv2.imread(str(SOURCE))
    if image is None:
        raise FileNotFoundError(f"Imagem ausente: {SOURCE}")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL))
    result = model.predict(
        source=str(SOURCE),
        imgsz=IMGSZ,
        conf=CONF,
        iou=0.5,
        device=0,
        verbose=False
    )[0]

    height, width = image.shape[:2]
    descriptors = []
    groups = []
    detections = []

    # Agrupamento incremental experimental
    # Não representa identificação definitiva de SKUs.
    for box in result.boxes:
        coords = box.xyxy[0].cpu().tolist()

        x1 = int(coords[0])
        y1 = int(coords[1])
        x2 = int(coords[2])
        y2 = int(coords[3])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        descriptor = describe(crop)
        best_group = None
        best_score = -2.0

        for group_id, reference in enumerate(descriptors):
            score = similarity(descriptor, reference)
            if score > best_score:
                best_score = score
                best_group = group_id

        if best_group is None or best_score < SIMILARITY:
            best_group = len(descriptors)
            descriptors.append(descriptor)
            groups.append([])

        groups[best_group].append(
            [x1, y1, x2, y2]
        )

        detections.append({
            "group_id": best_group,
            "bbox": [x1, y1, x2, y2],
            "confidence": round(
                float(box.conf.item()), 4
            )
        })

    overlay = image.copy()
    output_image = image.copy()

    rng = np.random.default_rng(42)
    colors = rng.integers(
        40, 255, size=(max(len(groups), 1), 3)
    )

    for group_id, boxes in enumerate(groups):
        color = tuple(
            int(v) for v in colors[group_id]
        )

        mask = np.zeros(
            (height, width), dtype=np.uint8
        )

        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(
                mask, (x1, y1), (x2, y2),
                255, -1
            )
            cv2.rectangle(
                overlay, (x1, y1), (x2, y2),
                color, -1
            )
            cv2.rectangle(
                output_image, (x1, y1),
                (x2, y2), color, 2
            )
            cv2.putText(
                output_image,
                f"P{group_id + 1}",
                (x1, max(y1 - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2
            )

        cv2.imwrite(
            str(OUTPUT / f"mask_P{group_id + 1}.png"),
            mask
        )

    blended = cv2.addWeighted(
        overlay, 0.25, output_image, 0.75, 0
    )

    cv2.imwrite(
        str(OUTPUT / "patterns_visual.png"),
        blended
    )

    report = {
        "source": str(SOURCE),
        "model": str(MODEL),
        "image_size": [width, height],
        "method": "HSV histogram correlation",
        "similarity_threshold": SIMILARITY,
        "groups": len(groups),
        "detections": detections
    }

    with open(
        OUTPUT / "patterns.json", "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report, f, indent=2,
            ensure_ascii=False
        )

    print(f"Deteccoes: {len(detections)}")
    print(f"Grupos visuais: {len(groups)}")
    print(f"Resultados: {OUTPUT}")


if __name__ == "__main__":
    main()
