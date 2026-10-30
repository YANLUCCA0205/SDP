"""Rotas HTTP. O endpoint /analyze segue o contrato do TCC (seção 4.3)."""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import AnalyzeResponse, ShiftIn, ShiftOut
from app.application.analyze_image import AnalyzeRequest
from app.application.errors import InvalidImageError, ModelUnavailableError
from app.container import Container
from app.domain.shift import Shift
from app.infrastructure.csv_export import records_to_csv
from app.infrastructure.yolo_detector import YoloDetector

router = APIRouter()
MAX_LIST_LIMIT = 1000
CSV_MEDIA_TYPE = "text/csv"


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


def _detector_status(detector: YoloDetector | None) -> dict[str, Any] | None:
    if detector is None:
        return None
    return {
        "weights": detector.weights_path.name,
        "loaded": detector.is_available,
        "error": detector.load_error,
        "classes": sorted(detector.class_names),
    }


@router.get("/", summary="Verificação de funcionamento (health check)")
def health(container: ContainerDep) -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "SDP - Sistema de Detecção de Produtos",
        "model": _detector_status(container.product_detector),
        "people_model": _detector_status(container.people_detector),
        "cameras_configured": container.shelf_profiles.camera_ids,
    }


@router.post("/analyze", response_model=AnalyzeResponse, summary="Recebe uma imagem e retorna as detecções")
async def analyze(
    container: ContainerDep,
    file: UploadFile = File(...),
    camera_id: str | None = Query(None, description="Identificador da câmera; padrão SDP_DEFAULT_CAMERA_ID"),
) -> AnalyzeResponse:
    camera_id = camera_id or container.settings.default_camera_id
    max_bytes = container.settings.max_upload_bytes
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Arquivo maior que o limite de {max_bytes} bytes")
    request = AnalyzeRequest(
        camera_id=camera_id,
        filename=file.filename or "",
        content_type=file.content_type or "",
        data=data,
    )
    try:
        # A inferência é síncrona e pesada: executa em threadpool para não bloquear o event loop (limitação 8-d).
        result = await run_in_threadpool(container.analyze.execute, request)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AnalyzeResponse.from_result(result)


@router.get("/ruptures", summary="Lista eventos de ruptura")
def list_ruptures(
    container: ContainerDep,
    camera_id: str | None = None,
    only_open: bool = False,
    include_pending: bool = False,
    limit: int = Query(100, ge=1, le=MAX_LIST_LIMIT),
) -> list[dict[str, Any]]:
    return container.report.records(
        camera_id=camera_id, confirmed_only=not include_pending, open_only=only_open, limit=limit
    )


@router.get("/ruptures/export.csv", summary="Exporta os eventos de ruptura em CSV")
def export_ruptures(
    container: ContainerDep,
    camera_id: str | None = None,
    limit: int = Query(MAX_LIST_LIMIT, ge=1, le=10_000),
) -> Response:
    records = container.report.records(camera_id=camera_id, limit=limit)
    return Response(content=records_to_csv(records), media_type=CSV_MEDIA_TYPE)


@router.post("/shifts", response_model=ShiftOut, status_code=201, summary="Cadastra turno de um funcionário")
def create_shift(payload: ShiftIn, container: ContainerDep) -> ShiftOut:
    shift = container.shifts.add(Shift(**payload.model_dump()))
    return ShiftOut(**vars(shift))


@router.get("/shifts", response_model=list[ShiftOut], summary="Lista a escala cadastrada")
def list_shifts(container: ContainerDep) -> list[ShiftOut]:
    return [ShiftOut(**vars(shift)) for shift in container.shifts.list()]


@router.delete("/shifts/{shift_id}", status_code=204, summary="Remove um turno")
def delete_shift(shift_id: int, container: ContainerDep) -> Response:
    if not container.shifts.delete(shift_id):
        raise HTTPException(status_code=404, detail="Turno não encontrado")
    return Response(status_code=204)
