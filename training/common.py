"""Utilidades compartilhadas pelos scripts de treinamento e avaliação."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_YAML = PROJECT_DIR / "dataset" / "data.yaml"


def resolve_data_yaml(data_yaml: Path, output_dir: Path) -> Path:
    """Grava uma cópia do data.yaml com `path` absoluto.

    O Ultralytics resolve `path` relativo contra o diretório global de datasets (e não contra o YAML),
    então um caminho absoluto evita depender dessa configuração.
    """
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = (data_yaml.parent / config.get("path", ".")).resolve()
    config["path"] = str(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = output_dir / "data.resolved.yaml"
    resolved.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return resolved
