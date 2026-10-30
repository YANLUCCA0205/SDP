"""Ponto de entrada compatível com o TCC: uvicorn main:app --host 0.0.0.0 --port 5000"""
from app.api.app_factory import create_app

app = create_app()
