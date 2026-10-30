"""Avaliação do modelo treinado: precisão, recall, mAP@0.5 e mAP@0.5:0.95 (TCC, seção 2.6).

Exemplos:
    python training/test_model.py --weights models/best.pt --split test
    python training/test_model.py --weights models/best.pt --split val --device cpu

Os valores impressos são os calculados pelo Ultralytics sobre o split escolhido; nada é estimado aqui.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_DATA_YAML, PROJECT_DIR, resolve_data_yaml  # noqa: E402

DEFAULT_OUTPUT_DIR = PROJECT_DIR / "runs" / "eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia pesos YOLO11 nos conjuntos de validação/teste")
    parser.add_argument("--weights", type=Path, default=PROJECT_DIR / "models" / "best.pt")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001, help="limiar baixo é o padrão para calcular mAP")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def summarize(metrics, names: dict[int, str], split: str) -> dict:
    return {
        "split": split,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_map50_95": {
            names[class_id]: float(value)
            for class_id, value in zip(metrics.box.ap_class_index.tolist(), metrics.box.maps[metrics.box.ap_class_index].tolist())
        },
    }


def main() -> None:
    args = parse_args()
    for path, hint in ((args.weights, "Treine o modelo primeiro (training/train.py)."), (args.data, "")):
        if not path.is_file():
            sys.exit(f"Arquivo não encontrado: {path}\n{hint}")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    data_yaml = resolve_data_yaml(args.data, args.output)
    options = dict(
        data=str(data_yaml), split=args.split, imgsz=args.imgsz, conf=args.conf,
        batch=args.batch, project=str(args.output), name=args.split, exist_ok=True,
    )
    if args.device:
        options["device"] = args.device
    metrics = model.val(**options)

    summary = summarize(metrics, model.names, args.split)
    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output / f"metrics_{args.split}.json"
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nRelatório salvo em {report}")


if __name__ == "__main__":
    main()
