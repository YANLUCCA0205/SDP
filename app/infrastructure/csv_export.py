"""Exportação dos eventos para CSV com Pandas (TCC, seção 3.6), usando os nomes de coluna da seção 2.5-b."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

COLUMNS = {
    "id_ruptura": "ID_Ruptura",
    "camera_id": "Camera",
    "data": "Data",
    "h_inicio": "H_Inicio",
    "h_fim": "H_Fim",
    "tempo_ruptura_min": "Tempo_Ruptura_Min",
    "produto": "Produto",
    "categoria": "Categoria",
    "corredor": "Corredor",
    "funcionario": "Funcionario",
    "fluxo_de_cliente": "Fluxo_de_Cliente",
    "status": "Status",
}


def records_to_csv(records: Sequence[dict[str, Any]]) -> str:
    frame = pd.DataFrame(list(records), columns=list(COLUMNS)).rename(columns=COLUMNS)
    return frame.to_csv(index=False)
