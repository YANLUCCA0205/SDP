"""Composição das dependências (Dependency Injection manual) a partir das configurações."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from app.application.analyze_image import AnalyzeImage
from app.application.rupture_report import RuptureReport
from app.application.track_rupture import TrackRupture
from app.config import Settings
from app.domain.flow import FlowThresholds
from app.domain.ports import ShiftRepository
from app.infrastructure.file_store import FileImageStore
from app.infrastructure.opencv_codec import OpenCvCodec
from app.infrastructure.shelf_config import YamlShelfProfiles
from app.infrastructure.sql_repositories import (
    SqlEventRepository,
    SqlShiftRepository,
    create_session_factory,
    create_sqlite_engine,
)
from app.infrastructure.yolo_detector import YoloDetector
from app.infrastructure.visual_history import VisualHistory


@dataclass
class Container:
    settings: Settings
    analyze: AnalyzeImage
    report: RuptureReport
    shifts: ShiftRepository
    shelf_profiles: YamlShelfProfiles
    product_detector: YoloDetector
    people_detector: YoloDetector | None
    visual_history: VisualHistory

    def load_models(self) -> None:
        self.product_detector.load()
        if self.people_detector is not None:
            self.people_detector.load()


def build_container(settings: Settings) -> Container:
    settings.resolve(settings.data_dir).mkdir(parents=True, exist_ok=True)
    session_factory = create_session_factory(create_sqlite_engine(settings.effective_database_url))
    events = SqlEventRepository(session_factory)
    shifts = SqlShiftRepository(session_factory)
    profiles = YamlShelfProfiles.from_file(settings.resolve(settings.shelf_config_path), settings.min_occupancy)
    product_detector = _product_detector(settings)
    people_detector = _people_detector(settings)
    tracker = TrackRupture(events, shifts, settings.confirmations_required)
    timezone = ZoneInfo(settings.timezone)
    visual_history = VisualHistory(settings.resolve(settings.data_dir) / 'references', product_detector,
                                   confirmations=settings.confirmations_required)
    analyze = AnalyzeImage(
        products=product_detector,
        people=people_detector,
        codec=OpenCvCodec(),
        store=FileImageStore(settings.resolve(settings.received_dir), settings.resolve(settings.results_dir)),
        profiles=profiles,
        tracker=tracker,
        clock=lambda: datetime.now(timezone).replace(tzinfo=None),
        max_bytes=settings.max_upload_bytes,
        visual_history=visual_history,
    )
    thresholds = FlowThresholds(settings.flow_low_max, settings.flow_medium_max)
    return Container(
        settings=settings,
        analyze=analyze,
        report=RuptureReport(events, thresholds),
        shifts=shifts,
        shelf_profiles=profiles,
        product_detector=product_detector,
        people_detector=people_detector,
        visual_history=visual_history,
    )


def _product_detector(settings: Settings) -> YoloDetector:
    return YoloDetector(
        settings.resolve(settings.model_path),
        confidence=settings.confidence,
        imgsz=settings.imgsz,
        device=settings.device,
    )


def _people_detector(settings: Settings) -> YoloDetector | None:
    if not settings.people_counting_enabled:
        return None
    return YoloDetector(
        settings.resolve(settings.person_model_path),
        confidence=settings.person_confidence,
        imgsz=settings.imgsz,
        device=settings.device,
        class_ids=[settings.person_class_id],
    )
