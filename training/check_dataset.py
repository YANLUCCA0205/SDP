"""Valida a estrutura e as anotações do dataset (formato YOLO de detecção) antes do treino.

Verifica, para train/val/test: pares imagem/rótulo, formato `class x_center y_center width height`
(valores normalizados entre 0 e 1) e índices de classe compatíveis com `names` do data.yaml.

Uso: python training/check_dataset.py [--data dataset/data.yaml] [--require-splits train val test]
Código de saída 0 = dataset válido; 1 = problemas encontrados.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATA = Path(__file__).resolve().parent.parent / "dataset" / "data.yaml"
ALL_SPLITS = ("train", "val", "test")
FIELDS_PER_LABEL = 5


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)
    images: dict[str, int] = field(default_factory=dict)
    instances: dict[int, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems


def check_label_line(line: str, class_count: int) -> str | None:
    parts = line.split()
    if len(parts) != FIELDS_PER_LABEL:
        return f"esperados {FIELDS_PER_LABEL} campos, encontrados {len(parts)}"
    try:
        class_id = int(parts[0])
        values = [float(v) for v in parts[1:]]
    except ValueError:
        return "valores não numéricos"
    if not 0 <= class_id < class_count:
        return f"classe {class_id} fora do intervalo 0..{class_count - 1}"
    if any(not 0.0 <= v <= 1.0 for v in values):
        return "coordenadas devem estar normalizadas entre 0 e 1"
    if values[2] <= 0 or values[3] <= 0:
        return "largura e altura devem ser maiores que zero"
    return None


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def check_split(root: Path, split: str, class_count: int, report: Report) -> None:
    images = image_files(root / "images" / split)
    report.images[split] = len(images)
    if not images:
        report.problems.append(f"[{split}] nenhuma imagem em images/{split}")
    labels_dir = root / "labels" / split
    for image in images:
        check_image_label(image, labels_dir / f"{image.stem}.txt", split, class_count, report)
    orphans = {p.stem for p in labels_dir.glob("*.txt")} - {i.stem for i in images} if labels_dir.is_dir() else set()
    for stem in sorted(orphans):
        report.problems.append(f"[{split}] rótulo sem imagem: {stem}.txt")


def check_image_label(image: Path, label: Path, split: str, class_count: int, report: Report) -> None:
    if not label.is_file():
        report.problems.append(f"[{split}] rótulo ausente para {image.name}")
        return
    # Arquivo de rótulo vazio é válido: imagem de fundo, sem objetos.
    for number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        error = check_label_line(line, class_count)
        if error:
            report.problems.append(f"[{split}] {label.name}:{number}: {error}")
        else:
            class_id = int(line.split()[0])
            report.instances[class_id] = report.instances.get(class_id, 0) + 1


def check_dataset(data_yaml: Path, splits: tuple[str, ...]) -> tuple[Report, dict[int, str]]:
    report = Report()
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = {int(k): str(v) for k, v in (config.get("names") or {}).items()}
    if not names:
        report.problems.append("data.yaml sem `names`")
        return report, names
    root = (data_yaml.parent / config.get("path", ".")).resolve()
    for split in splits:
        check_split(root, split, len(names), report)
    return report, names


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida o dataset YOLO do SDP")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--require-splits", nargs="+", choices=ALL_SPLITS, default=list(ALL_SPLITS))
    args = parser.parse_args()
    report, names = check_dataset(args.data, tuple(args.require_splits))

    print("Imagens por conjunto:", report.images)
    print("Instâncias por classe:", {names.get(k, k): v for k, v in sorted(report.instances.items())})
    for problem in report.problems:
        print("PROBLEMA:", problem)
    print("Dataset válido." if report.ok else f"{len(report.problems)} problema(s) encontrado(s).")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
