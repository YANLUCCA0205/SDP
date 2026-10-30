
from pathlib import Path
import cv2
import json

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "testes_reais/gondola_petyan.jpg"
OUTPUT = ROOT / "runs/shelf_analysis/petyan"



def main():
    image = cv2.imread(str(SOURCE))

    if image is None:
        raise FileNotFoundError(SOURCE)

    height, width = image.shape[:2]
    # Limites verticais aproximados da fotografia
    # Ajustaremos após visualizar o resultado.
    # Regioes aproximadas na imagem de referencia 928x1648
    SHELVES_REFERENCE = [
        (130, 290),
        (320, 510),
        (540, 710),
        (750, 925),
        (970, 1205),
        (1240, 1495),
    ]

    # Converte para a altura real da fotografia
    SHELVES = [
        (
            round(y1 * height / 1648),
            round(y2 * height / 1648)
        )
        for y1, y2 in SHELVES_REFERENCE
    ]
    print(f"Dimensoes da imagem: {width} x {height}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    visual = image.copy()
    shelves = []

    for index, (y1, y2) in enumerate(SHELVES, 1):
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        if y2 <= y1:
            continue

        cv2.rectangle(
            visual,
            (0, y1),
            (width - 1, y2),
            (0, 255, 255),
            2
        )

        cv2.putText(
            visual,
            f"PRATELEIRA {index}",
            (15, min(y1 + 30, height - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )

        shelves.append({
            "id": index,
            "y1": y1,
            "y2": y2
        })

    cv2.imwrite(
        str(OUTPUT / "shelves_visual.png"),
        visual
    )

    with open(
        OUTPUT / "shelves.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump({
            "source": str(SOURCE),
            "image_size": [width, height],
            "shelves": shelves
        }, f, indent=2)

    print(f"Prateleiras mapeadas: {len(shelves)}")
    print(f"Resultados: {OUTPUT}")


if __name__ == "__main__":
    main()
