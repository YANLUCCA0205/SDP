"""Erros de aplicação, traduzidos para HTTP pela camada de API."""


class InvalidImageError(Exception):
    """Entrada inválida do cliente (HTTP 400): tipo, tamanho, conteúdo ou identificador de câmera."""


class ModelUnavailableError(Exception):
    """Pesos do modelo ausentes ou não carregados (HTTP 503)."""
