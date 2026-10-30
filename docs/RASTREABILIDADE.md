# Matriz de rastreabilidade — TCC → implementação → teste

Legenda de situação: **T** = testado e executado com sucesso neste ambiente · **E** = implementado, testes escritos,
mas **não executados** (dependências indisponíveis) · **U** = depende de você (dados/treino/execução) · **—** = fora do backend.

| ID | Requisito (seção do TCC) | Módulo | Teste | Situação |
|---|---|---|---|---|
| RF01 | `GET /` health check (4.3) | `api/routes.py::health` | `test_api::test_health_reports_model_not_loaded` | E |
| RF02 | `POST /analyze`, multipart campo `file`, só `image/jpeg`/`image/png` (4.3, 6.1) | `api/routes.py::analyze`, `application/analyze_image.py` | `test_analyze_image::test_rejects_unsupported_content_type`; `test_api::test_unsupported_*` | T (aplicação) / E (HTTP) |
| RF03 | Respostas 200 e 400 com `detail` (4.3) | `api/routes.py`, `api/schemas.py` | `test_analyze_image::test_rejects_*`; `test_api::test_analyze_success_matches_tcc_contract` | T (400 na aplicação) / E (contrato HTTP) |
| RF04 | Detecção com `class_id`, `class_name`, `confidence` (4 casas), `bbox` x1,y1,x2,y2 (4.3, 6.2) | `infrastructure/yolo_detector.py`, `domain/detection.py` | `test_yolo_integration`; `test_api::test_analyze_success_*` | E |
| RF05 | `IMREAD_COLOR`, `conf=0.25`, modelo carregado uma vez (4.4, 5.2) | `opencv_codec.py`, `config.py`, `yolo_detector.py::load`, `app_factory.py` (lifespan) | `test_analyze_image::test_rejects_undecodable_image` | T (decodificação) / E (YOLO) |
| RF06 | Gravar imagem recebida e anotada (4.4) | `file_store.py`, `opencv_codec.annotate` | `test_analyze_image::test_stores_unique_history_and_latest_copies_per_camera` | T |
| RF07 | Trocar `yolo11n.pt` → `best.pt` sem mexer no firmware (2.1, 4.2) | `config.py` (`SDP_MODEL_PATH`), `ports.ObjectDetector` | `test_report_and_config::test_missing_weights_*`; `test_analyze_image::test_model_unavailable_*` | T (comportamento sem pesos) / U (best.pt) |
| RF08 | Regra: contagem por classe vs. esperado → disponível / possível ruptura (2.7, 4.5) + 60% da capacidade (sua definição) | `domain/shelf.py` | `test_domain::ShelfProfileTest`, `ProductSlotTest` | T |
| RF09 | Início = 1ª captura em ruptura; fim = 1ª captura disponível; duração (2.7) | `domain/rupture.py`, `application/track_rupture.py` | `test_domain::RuptureEventTest`; `test_track_rupture::test_full_lifecycle_*` | T |
| RF10 | Confirmar só após mais de uma captura consecutiva (2.7) | `RuptureEvent.register_capture`, `SDP_CONFIRMATIONS_REQUIRED` | `test_track_rupture::test_pending_event_is_discarded_*`; `test_domain::test_starts_pending_*` | T |
| RF11 | Registro com ID, data, H_Inicio, H_Fim, Tempo_Ruptura_Min, produto, categoria, corredor, funcionário, fluxo (2.5-b) | `sql_repositories.py`, `application/rupture_report.py` | `test_report_and_config::RuptureReportTest`; `test_sql_repositories` | T (relatório) / E (SQLite) |
| RF12 | Responsável via cruzamento horário × escala, sem reconhecimento facial (2.4) | `domain/shift.py`, `TrackRupture._responsible_at`, `/shifts` | `test_track_rupture::test_*employee*`, `test_domain::ShiftTest` | T (regra) / E (SQLite/API) |
| RF13 | Fine-tuning de `yolo11n.pt`, 640 px, ~100 épocas → `best.pt` (2.6) | `training/train.py`, `dataset/data.yaml` | — | E / U (executar o treino) |
| RF14 | Avaliação: precisão, recall, mAP@0.5, mAP@0.5:0.95 em val/test (2.6) | `training/test_model.py` | — | E / U |
| RF15 | Estrutura-alvo: `dataset/{images,labels}/{train,val,test}`, `data.yaml`, `train.py`, `test_model.py` (5.1) | árvore do projeto; `training/check_dataset.py` | `test_check_dataset` | T (validador) |
| RF16 | Execução `uvicorn main:app --host 0.0.0.0 --port 5000` (3.3, 6.5) | `main.py`, `README.md` | — | E |
| RF17 | Pandas para organizar/exportar eventos (3.6) | `infrastructure/csv_export.py`, `GET /ruptures/export.csv` | `test_report_and_config::CsvExportTest` | T |
| RNF-a | Reprodutibilidade: `requirements.txt`, Python 3.11, `.gitignore`, README (2.1, 7.1) | raiz | — | T (arquivos) / E (instalação) |
| RNF-b | Desacoplamento firmware / API / inferência / regra (2.1, 4.2) | `domain/ports.py`, camadas | testes de regra sem modelo | T |
| RNF-c | LGPD sobre imagens de trabalhadores (2.4) | sem reconhecimento facial; aviso de retenção no README | — | parcial (política de retenção é sua) |
| RNF-d | Validação de entradas inválidas (6.1) | `AnalyzeImage._validate/_decode` | `test_analyze_image` | T |
| RNF-e | Testes automatizados (limitação 8-h) | `tests/` | 64 executados, 18 pulados | T / E |
| LIM-c | Sobrescrita/condição de corrida (8-c) | `file_store.py` (nome único + troca atômica) | `test_stores_unique_history_*` | T |
| LIM-d | Inferência bloqueante (8-d) | `run_in_threadpool` em `routes.py`; *lock* no detector | — | E |
| LIM-f | PNG salvo como `.jpg` (8-f) | `ALLOWED_CONTENT_TYPES` | `test_png_is_stored_with_png_extension` | T |
| EXT-1 | Várias câmeras, cada uma em uma gôndola | `camera_id`, `config/shelves.yaml`, pastas por câmera | `test_cameras_are_stored_separately` | T |
| EXT-2 | Fluxo de clientes = pessoas no raio da câmera (sua definição) | `YoloDetector(class_ids=[0])`, `domain/flow.py` | `test_analyze_image::*people*`, `test_domain::FlowThresholdsTest` | T (lógica) / E (YOLO real) |
| EXT-3 | Docker (sua solicitação; TCC 3.6 o prevê) | `Dockerfile`, `docker-compose*.yml` | — | escrito, **não construído** |

## Fora do backend / não atendido

- **Firmware** (8-e credenciais fixas, 8-g descarte dos primeiros quadros): `conf-cam.ino` não foi alterado.
- **HTTPS e autenticação** (8-e): não implementados; o TCC os considera aceitáveis em rede local de laboratório.
- **Análises da seção 2.8** (mineração sequencial/temporal/espacial/de processos) e **impacto financeiro** (2.9): são
  técnicas de análise do trabalho acadêmico, não requisitos do backend. A exportação CSV entrega os dados para elas.
- **Métricas do modelo**: inexistentes até o treino.
