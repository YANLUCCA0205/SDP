"""Criação da aplicação FastAPI."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings
from app.container import Container, build_container

DESCRIPTION = (
    "API do SDP: recebe imagens de gôndolas (ESP32-CAM), detecta produtos com YOLO11 e registra "
    "eventos de ruptura de estoque."
)


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or (container.settings if container else Settings())
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active = container or build_container(settings)
        try:
            active.load_models()  # o modelo é carregado uma única vez, na inicialização (TCC 5.2)
            app.state.container = active
            yield
        finally:
            # No Windows, conexões SQLite abertas impedem a remoção de bancos temporários.
            if active.db_engine is not None:
                active.db_engine.dispose()

    app = FastAPI(title="SDP - Detecção de Produtos", description=DESCRIPTION, version="1.0.0", lifespan=lifespan)
    app.include_router(router)
    return app
