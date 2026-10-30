
from pathlib import Path
import json

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

IMAGE = ROOT / "testes_reais/gondola_petyan.jpg"
MODEL = ROOT / "runs/train/teste_inicial/weights/best.pt"
SHELVES_JSON = ROOT / "runs/shelf_analysis/petyan/shelves.json"
OUTPUT = ROOT / "runs/occupancy/petyan"

CONF = 0.25
IMGSZ = 1280
MIN_GAP_WIDTH = 35
MIN_OVERLAP = 0.50


def intersection_area(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h


def vertical_overlap(a, b):
    overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller_height = min(a[3] - a[1], b[3] - b[1])
    return overlap / max(smaller_height, 1)


def main():
    image = cv2.imread(str(IMAGE))
    if image is None:
        raise FileNotFoundError(IMAGE)

    height, width = image.shape[:2]

    with open(SHELVES_JSON, encoding="utf-8") as f:
        config = json.load(f)

    if config["image_size"] != [width, height]:
        raise ValueError("Dimensoes do mapa e da imagem diferentes")

    model = YOLO(str(MODEL))
    result = model.predict(
        source=str(IMAGE),
        imgsz=IMGSZ,
        conf=CONF,
        iou=0.5,
        device=0,
        verbose=False
    )[0]

    detections = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(
            int, box.xyxy[0].cpu().tolist()
        )

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": float(box.conf.item())
        })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    visual = image.copy()
    report = []

    for shelf in config["shelves"]:
        sid = shelf["id"]
        top, bottom = shelf["y1"], shelf["y2"]

        region = [0, top, width, bottom]
        region_height = bottom - top

        # Aloca uma deteccao na prateleira com
        # maior intersecao com sua caixa.
        assigned = []

        for det in detections:
            box = det["bbox"]
            box_area = (box[2]-box[0]) * (box[3]-box[1])

            if box_area <= 0:
                continue

            current_area = intersection_area(box, region)

            all_areas = [
                intersection_area(
                    box,
                    [0, s["y1"], width, s["y2"]]
                )
                for s in config["shelves"]
            ]

            if (
                current_area > 0
                and current_area == max(all_areas)
                and all_areas.index(max(all_areas))
                    == sid - 1
            ):
                assigned.append(det)

        # Mascara binaria: uniao das caixas detectadas
        mask = np.zeros(
            (region_height, width), dtype=np.uint8
        )

        for det in assigned:
            x1, y1, x2, y2 = det["bbox"]
            ly1 = max(0, y1 - top)
            ly2 = min(region_height, y2 - top)

            if ly2 > ly1:
                cv2.rectangle(
                    mask,
                    (x1, ly1),
                    (x2, ly2),
                    255,
                    -1
                )

            cv2.rectangle(
                visual, (x1, y1), (x2, y2),
                (0, 220, 0), 2
            )

        coverage = float(np.count_nonzero(mask)) / max(
            width * region_height, 1
        )

        gaps = []
        ordered = sorted(
            assigned, key=lambda d: d["bbox"][0]
        )

        for i, a in enumerate(ordered):
            a_box = a["bbox"]

            for b in ordered[i+1:]:
                b_box = b["bbox"]

                if vertical_overlap(a_box, b_box) < MIN_OVERLAP:
                    continue

                gap_width = b_box[0] - a_box[2]

                if gap_width < MIN_GAP_WIDTH:
                    continue

                gy1 = max(a_box[1], b_box[1], top)
                gy2 = min(a_box[3], b_box[3], bottom)

                gap_box = [a_box[2], gy1, b_box[0], gy2]

                if gy2 <= gy1:
                    continue

                # Evita sinalizar regioes ocupadas
                # por outras caixas detectadas.
                gap_area = intersection_area(
                    gap_box, gap_box
                )

                occupied = any(
                    intersection_area(
                        gap_box, d["bbox"]
                    ) / max(gap_area, 1) > 0.20
                    for d in assigned
                )

                if occupied:
                    continue

                gaps.append(gap_box)
                break

        # Deduplicacao dos candidatos
        unique_gaps = []
        for gap in gaps:
            if not any(
                intersection_area(gap, old) /
                max(intersection_area(gap, gap), 1) > 0.5
                for old in unique_gaps
            ):
                unique_gaps.append(gap)

        for x1, y1, x2, y2 in unique_gaps:
            cv2.rectangle(
                visual,
                (x1, y1), (x2, y2),
                (0, 165, 255), 3
            )

        cv2.rectangle(
            visual,
            (0, top), (width-1, bottom),
            (255, 255, 0), 3
        )

        cv2.putText(
            visual,
            f"P{sid} | {len(assigned)} objetos | "
            f"Cobertura: {coverage:.1%}",
            (20, min(top + 45, height-10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 255), 2
        )

        cv2.imwrite(
            str(OUTPUT / f"mask_shelf_{sid}.png"),
            mask
        )

        report.append({
            "shelf_id": sid,
            "detections": len(assigned),
            "coverage_percent": round(
                coverage * 100, 2
            ),
            "suspected_gaps": unique_gaps,
            "gap_count": len(unique_gaps)
        })

    cv2.imwrite(
        str(OUTPUT / "occupancy_visual.png"),
        visual
    )

    with open(
        OUTPUT / "occupancy.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump({
            "source": str(IMAGE),
            "method": "bbox_occupancy_v1",
            "warning": (
                "Cobertura visual e lacunas nao "
                "comprovam ruptura de estoque."
            ),
            "shelves": report
        }, f, indent=2, ensure_ascii=False)

    print("\nRESULTADOS DA ANALISE")
    for shelf in report:
        print(
            f"Prateleira {shelf['shelf_id']}: "
            f"{shelf['detections']} deteccoes | "
            f"{shelf['coverage_percent']}% cobertura | "
            f"{shelf['gap_count']} vaos candidatos"
        )

    print(f"\nArquivos salvos em: {OUTPUT}")


if __name__ == "__main__":
    main()