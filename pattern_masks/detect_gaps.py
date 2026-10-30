from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/pattern_masks/foto_1/patterns.json"
OUTPUT = ROOT / "runs/ruptures/foto_1"

# Parâmetros iniciais, sujeitos a calibração
MIN_GAP_RATIO = 0.55
MAX_GAP_RATIO = 2.5
MIN_VERTICAL_OVERLAP = 0.60
MIN_CONFIDENCE = 0.35


def vertical_overlap(a, b):
    top = max(a[1], b[1])
    bottom = min(a[3], b[3])
    intersection = max(0, bottom - top)

    height_a = a[3] - a[1]
    height_b = b[3] - b[1]

    return intersection / max(
        min(height_a, height_b), 1
    )


def occupied_by_other(gap, detections, exclude):
    """Verifica se outro objeto ocupa a região suspeita."""
    gx1, gy1, gx2, gy2 = gap
    area = max((gx2 - gx1) * (gy2 - gy1), 1)

    for index, item in enumerate(detections):
        if index in exclude:
            continue

        x1, y1, x2, y2 = item["bbox"]

        iw = max(0, min(gx2, x2) - max(gx1, x1))
        ih = max(0, min(gy2, y2) - max(gy1, y1))

        if (iw * ih) / area > 0.30:
            return True

    return False


def main():
    with open(INPUT, encoding="utf-8") as f:
        data = json.load(f)

    source = Path(data["source"])
    image = cv2.imread(str(source))

    if image is None:
        raise FileNotFoundError(
            f"Imagem original nao encontrada: {source}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    detections = [
        d for d in data["detections"]
        if d["confidence"] >= MIN_CONFIDENCE
    ]

    groups = {}

    for index, item in enumerate(detections):
        group_id = item["group_id"]
        groups.setdefault(group_id, []).append(
            (index, item)
        )

    candidates = []

    for group_id, objects in groups.items():
        objects.sort(
            key=lambda entry: entry[1]["bbox"][0]
        )

        for i in range(len(objects) - 1):
            index_a, item_a = objects[i]
            a = item_a["bbox"]

            for j in range(i + 1, len(objects)):
                index_b, item_b = objects[j]
                b = item_b["bbox"]

                # Somente embalagens alinhadas
                if vertical_overlap(a, b) < MIN_VERTICAL_OVERLAP:
                    continue

                width_a = a[2] - a[0]
                width_b = b[2] - b[0]
                typical_width = (width_a + width_b) / 2

                if typical_width <= 0:
                    continue

                gap_width = b[0] - a[2]

                if gap_width <= 0:
                    continue

                ratio = gap_width / typical_width

                if not (
                    MIN_GAP_RATIO <= ratio <= MAX_GAP_RATIO
                ):
                    continue

                # Usa a faixa vertical comum às embalagens
                gy1 = max(a[1], b[1])
                gy2 = min(a[3], b[3])

                gap = [
                    int(a[2]), int(gy1),
                    int(b[0]), int(gy2)
                ]

                if occupied_by_other(
                    gap, detections, {index_a, index_b}
                ):
                    continue

                candidates.append({
                    "group_id": int(group_id),
                    "bbox": gap,
                    "relative_gap": round(float(ratio), 3),
                    "status": "possivel_falta",
                    "method": "spatial_gap"
                })

    # Remove candidatos muito sobrepostos
    candidates.sort(
        key=lambda c: c["relative_gap"],
        reverse=True
    )

    selected = []

    for candidate in candidates:
        a = candidate["bbox"]
        duplicate = False

        for existing in selected:
            b = existing["bbox"]

            iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))

            intersection = iw * ih

            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])

            union = area_a + area_b - intersection
            iou = intersection / max(union, 1)

            if iou > 0.50:
                duplicate = True
                break

        if not duplicate:
            selected.append(candidate)

    overlay = image.copy()

    for candidate in selected:
        x1, y1, x2, y2 = candidate["bbox"]

        cv2.rectangle(
            overlay,
            (x1, y1), (x2, y2),
            (0, 200, 255), -1
        )

    visual = cv2.addWeighted(
        overlay, 0.30, image, 0.70, 0
    )

    for candidate in selected:
        x1, y1, x2, y2 = candidate["bbox"]

        cv2.rectangle(
            visual,
            (x1, y1), (x2, y2),
            (0, 170, 255), 2
        )

        cv2.putText(
            visual, "POSSIVEL FALTA",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40, (0, 120, 255), 1
        )

    cv2.imwrite(
        str(OUTPUT / "gaps_visual.png"),
        visual
    )

    report = {
        "source": str(source),
        "method": "spatial_gap_v1",
        "candidate_count": len(selected),
        "candidates": selected,
        "warning": (
            "Candidatos espaciais nao comprovam "
            "ruptura de estoque."
        )
    }

    with open(
        OUTPUT / "gaps.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            report, f, indent=2,
            ensure_ascii=False
        )

    print(f"Objetos analisados: {len(detections)}")
    print(f"Grupos analisados: {len(groups)}")
    print(f"Vaos suspeitos: {len(selected)}")
    print(f"Resultados: {OUTPUT}")


if __name__ == "__main__":
    main()
