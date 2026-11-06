"""Configuração centralizada por variáveis de ambiente (prefixo SDP_) ou arquivo .env."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
BYTES_PER_MB = 1024 * 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_prefix="SDP_", extra="ignore", protected_namespaces=()
    )

    # Modelo. Trocar yolo11n.pt (provisório) por best.pt = apontar SDP_MODEL_PATH para os pesos treinados.
    model_path: Path = Path("models/best.pt")
    confidence: float = Field(default=0.25, ge=0, le=1)  # TCC 6.2
    imgsz: int = Field(default=640, gt=0)
    device: str | None = None  # "cpu", "0", "cuda:0"... None = automático

    # Fluxo de clientes: pessoas detectadas por um modelo COCO (classe 0 = person).
    person_model_path: Path = Path("models/yolo11n.pt")
    person_class_id: int = 0
    person_confidence: float = Field(default=0.25, ge=0, le=1)
    people_counting_enabled: bool = True

    # Regra de ruptura.
    min_occupancy: float = Field(default=0.6, gt=0, le=1)  # gôndola com 60% da capacidade
    confirmations_required: int = Field(default=2, ge=1)  # TCC 2.7: mais de uma captura consecutiva
    flow_low_max: float = Field(default=1.0, ge=0)  # média de pessoas por captura
    flow_medium_max: float = Field(default=3.0, ge=0)
    shelf_config_path: Path = Path("config/shelves.yaml")
    default_camera_id: str = "cam-01"  # usado quando o firmware não informa camera_id

    # Armazenamento.
    data_dir: Path = Path("data")
    received_dir: Path = Path("received")
    results_dir: Path = Path("results")
    database_url: str | None = None  # padrão: sqlite em <data_dir>/sdp.db
    max_upload_mb: float = Field(default=5.0, gt=0)
    timezone: str = "America/Sao_Paulo"

    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * BYTES_PER_MB)

    @staticmethod
    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def effective_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.resolve(self.data_dir) / 'sdp.db'}"
