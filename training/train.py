"""Fine-tuning do YOLO11 a partir de yolo11n.pt (TCC, seção 2.6).

Exemplos:
    python training/train.py                          # padrões do TCC: imgsz 640, 100 épocas
    python training/train.py --device 0 --batch 32    # GPU
    python training/train.py --device cpu --batch 8   # CPU
    python training/train.py --install                # copia o best.pt resultante para models/best.pt
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_DATA_YAML, PROJECT_DIR, resolve_data_yaml  # noqa: E402
from check_dataset import check_dataset  # noqa: E402

DEFAULT_BASE_WEIGHTS = PROJECT_DIR / "models" / "yolo11n.pt"
DEFAULT_RUNS_DIR = PROJECT_DIR / "runs" / "train"
INSTALL_TARGET = PROJECT_DIR / "models" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o YOLO11 com o dataset do SDP")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML, help="data.yaml do dataset")
    parser.add_argument("--weights", type=Path, default=DEFAULT_BASE_WEIGHTS, help="pesos iniciais (transfer learning)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="use -1 para ajuste automático em GPU")
    parser.add_argument("--device", default=None, help="'cpu', '0', '0,1'... (padrão: automático)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30, help="épocas sem melhora antes da parada antecipada")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--name", default="sdp")
    parser.add_argument("--install", action="store_true", help=f"copia best.pt para {INSTALL_TARGET}")
    return parser.parse_args()


def require_file(path: Path, hint: str) -> None:
    if not path.is_file():
        sys.exit(f"Arquivo não encontrado: {path}\n{hint}")


def main() -> None:
    args = parse_args()
    require_file(args.data, "Crie o dataset conforme o README (seção de treinamento).")
    report, names = check_dataset(args.data, ("train", "val", "test"))
    if not report.ok:
        sys.exit("Dataset invalido. Rode python training/check_dataset.py:\n" + "\n".join(report.problems))
    if sum(report.instances.values()) == 0:
        sys.exit("Dataset sem objetos anotados: nao treinar com labels vazias. Anote as fotos e confira data.yaml.")
    require_file(args.weights, "Baixe o yolo11n.pt para models/ (veja scripts/download_weights.py).")

    from ultralytics import YOLO

    data_yaml = resolve_data_yaml(args.data, args.project / args.name)
    train_options = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )
    if args.device:
        train_options["device"] = args.device
    YOLO(str(args.weights)).train(**train_options)

    best = args.project / args.name / "weights" / "best.pt"
    print(f"\nPesos finais: {best}")
    if args.install:
        require_file(best, "O treino não gerou best.pt.")
        shutil.copy2(best, INSTALL_TARGET)
        print(f"Copiado para {INSTALL_TARGET}. Use SDP_MODEL_PATH=models/best.pt para ativá-lo.")


if __name__ == "__main__":
    main()
