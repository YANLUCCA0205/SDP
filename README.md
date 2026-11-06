# SDP — Sistema de Detecção de Produtos

## Atualização experimental - 06/11/2026

O detector de produtos usa `models/best.pt` treinado para a classe genérica
`object`. Os agrupamentos visuais são gerados **depois** da inferência por
`app/infrastructure/visual_history.py` e acionados pelo caso de uso
`app/application/analyze_image.py`.

Fluxo: ESP32-CAM -> API -> YOLO -> referência por câmera -> padrões visuais
por prateleira -> comparação espacial -> confirmação em 2 capturas distintas ->
evento de ausência / reposição.

1. Inicie a API com `uvicorn main:app --host 0.0.0.0 --port 5000`.
2. Cadastre uma foto de referência com `POST /reference?camera_id=cam-01`.
   **Cadastrar nova referência reinicializa os estados temporais dessa câmera.**
3. Envie capturas em **ordem cronológica** a `POST /analyze?camera_id=cam-01`.
4. O campo `ruptura_visual` diferencia `ok`, `possivel_ruptura`,
   `ruptura` e `indeterminado`; `eventos` lista confirmação e retorno.

A decisão é relativa à disposição visível na referência, não ao estoque total
da loja nem a identificação comercial/SKU. A funcionalidade temporal revisada
ainda precisa ser executada e calibrada no notebook com as 13 imagens reais.
Um ensaio sintético com detector simulado demonstrou os estados
presença -> retirada -> confirmação -> reposição; **isso não é precisão do YOLO**.

**Atenção:** as rotas legadas de `/ruptures` e o banco SQLite de turnos ainda
não foram unificados com o registro temporal JSON de `data/references`.
Não misture as fontes como se fossem os mesmos eventos.


Backend do TCC *Sistemas de Detecção de Produtos Utilizando Visão Computacional e Redes Neurais Convolucionais*
(Universidade de Uberaba). Câmeras ESP32-CAM enviam imagens de gôndolas por HTTP; a API detecta produtos com
**Ultralytics YOLO11**, aplica a regra de **ruptura de gôndola** e registra eventos (início, fim, duração, corredor,
funcionário responsável e fluxo de clientes) para análise gerencial.

> **Estado desta entrega**: veja a seção [Verificação](#verificação-o-que-foi-e-o-que-não-foi-executado). O modelo
> **não foi treinado** (depende do seu dataset) e as métricas de acurácia só existirão após o treino.

## Sumário
1. [Arquitetura](#arquitetura) · 2. [Regras de negócio](#regras-de-negócio) · 3. [Instalação](#instalação)
4. [Configuração](#configuração) · 5. [Executar](#executar) · 6. [API](#api) · 7. [Treinamento](#treinamento-do-yolo11)
8. [Testes](#testes) · 9. [Docker](#docker) · 10. [Firmware](#firmware-esp32-cam) · 11. [Problemas comuns](#problemas-comuns)
12. [Verificação](#verificação-o-que-foi-e-o-que-não-foi-executado) · 13. [Limitações](#limitações-e-decisões-técnicas)

## Arquitetura

Camadas com *Ports and Adapters* enxuto. O domínio não conhece FastAPI, Ultralytics nem SQL; isso permite trocar
`yolo11n.pt` por `best.pt` só por configuração e testar a regra de ruptura sem modelo (TCC 4.2: "detecção separada da
decisão de negócio").

```
SDP/
├── main.py                    # uvicorn main:app
├── app/
│   ├── config.py              # Settings (variáveis SDP_*)
│   ├── container.py           # composição das dependências
│   ├── domain/                # puro: Detection, ShelfProfile/ShelfState, RuptureEvent, Shift, FlowThresholds, ports
│   ├── application/           # AnalyzeImage, TrackRupture, RuptureReport
│   ├── infrastructure/        # YoloDetector, OpenCvCodec, FileImageStore, SQLite (SQLAlchemy), YAML, CSV (Pandas)
│   └── api/                   # rotas, schemas, app factory
├── training/                  # train.py, test_model.py, check_dataset.py
├── dataset/                   # images|labels × train|val|test + data.yaml
├── config/shelves.yaml        # gôndola (corredor, produtos, capacidade) por câmera
├── models/                    # yolo11n.pt (provisório), best.pt (treinado)
├── tests/  scripts/  esp32-cam/  Dockerfile  docker-compose*.yml
```

Fluxo de `POST /analyze`: valida → decodifica (OpenCV, 3 canais) → YOLO de produtos → YOLO de pessoas (fluxo de
clientes) → grava imagens → avalia a gôndola da câmera → atualiza eventos de ruptura → responde JSON. A inferência roda em
*threadpool* (não bloqueia o servidor) e o acesso ao modelo é serializado por *lock*.

**Exceções ao Object Calisthenics** (idiomaticidade Python): `dataclass` com atributos públicos para *value objects*
e modelos Pydantic; uso de `else`/`if` sequenciais curtos onde *early return* é mais legível.

## Regras de negócio

| Regra | Origem |
|---|---|
| YOLO só detecta; a ruptura é decidida em camada posterior | TCC 2.7, 4.2 |
| Possível ruptura quando `detectado < ceil(capacidade × 0,6)` por produto | TCC 2.7 + sua definição ("gôndola com 60% da capacidade") |
| Início = 1ª captura em ruptura; fim = 1ª captura disponível; duração = fim − início | TCC 2.7 |
| Evento só é **confirmado** após `SDP_CONFIRMATIONS_REQUIRED` (padrão 2) capturas consecutivas; pendente que se recupera é descartado | TCC 2.7 |
| Responsável = funcionário(s) escalado(s) no corredor no instante do **fim** do evento. Sem reconhecimento facial | TCC 2.4 |
| Fluxo de clientes = pessoas dentro do raio da câmera (classe `person` do COCO). Nível baixo/médio/alto sobre a **média de pessoas por captura** durante o evento | sua definição |
| Produtos do perfil que o modelo carregado **não conhece** não são avaliados (evita ruptura falsa com `yolo11n.pt`) | decisão técnica |

> **Interpretação a confirmar**: "60% da capacidade" foi implementada como mínimo por produto
> (`capacity` em `config/shelves.yaml` = máximo de unidades visíveis com a gôndola cheia). Se a sua intenção for outra
> (ex.: ocupação por área da prateleira), a regra está isolada em `ProductSlot.required_units`.

## Instalação

Requisitos: **Python 3.11**, ~2 GB de disco (PyTorch). GPU NVIDIA é opcional.

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env
python scripts/download_weights.py # baixa yolo11n.pt (COCO) para models/ — exige internet
```

Em máquina sem GPU, instale o PyTorch para CPU antes (imagem bem menor):
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
As versões em `requirements.txt` são limites inferiores; após instalar, registre as que funcionaram com
`pip freeze > requirements.lock`.

## Configuração

Tudo por variáveis de ambiente (prefixo `SDP_`) ou `.env` — veja `.env.example`, comentado. As principais:

| Variável | Padrão | Função |
|---|---|---|
| `SDP_MODEL_PATH` | `models/yolo11n.pt` | Pesos dos produtos. **Trocar por `models/best.pt` após o treino** |
| `SDP_CONFIDENCE` | `0.25` | Limiar de confiança (TCC 6.2: reavaliar com o modelo treinado) |
| `SDP_DEVICE` | automático | `cpu`, `0`, `cuda:0` |
| `SDP_MIN_OCCUPANCY` | `0.6` | Fração mínima da capacidade |
| `SDP_CONFIRMATIONS_REQUIRED` | `2` | Capturas consecutivas para confirmar a ruptura |
| `SDP_PEOPLE_COUNTING_ENABLED` | `true` | Contagem de pessoas (fluxo de clientes) |
| `SDP_FLOW_LOW_MAX` / `SDP_FLOW_MEDIUM_MAX` | `1.0` / `3.0` | Limites do fluxo — **valores provisórios, calibre com dados reais** |
| `SDP_DEFAULT_CAMERA_ID` | `cam-01` | Câmera assumida quando o firmware não informa `camera_id` |
| `SDP_MAX_UPLOAD_MB` | `5` | Limite de tamanho do arquivo |
| `SDP_TIMEZONE` | `America/Sao_Paulo` | Fuso dos horários dos eventos e da escala |

**Gôndolas por câmera** — edite `config/shelves.yaml` (corredor, produtos, categoria, capacidade). Os `class_name`
precisam ser **idênticos** aos nomes em `dataset/data.yaml`. Os valores que vêm no arquivo são exemplos provisórios.

**Escala de funcionários** — cadastrada pela API (`POST /shifts`), sem CSV. Banco: SQLite em `data/sdp.db`, criado
automaticamente na primeira execução (tabelas via `create_all`; não há migrações).

## Executar

```bash
uvicorn main:app --host 0.0.0.0 --port 5000          # como no TCC; adicione --reload em desenvolvimento
```

Documentação interativa (OpenAPI/Swagger): `http://localhost:5000/docs`. Use **um único worker** (padrão): a
serialização por câmera da regra de ruptura é por processo. Libere a porta 5000 no firewall para a ESP32-CAM.

Sem os pesos, a API sobe normalmente, `GET /` informa `model.loaded=false` e `POST /analyze` responde **503**. Não
há *fallback* silencioso.

## API

| Método | Rota | Descrição |
|---|---|---|
| GET | `/` | Health check: estado dos modelos e câmeras configuradas |
| POST | `/analyze?camera_id=cam-01` | `multipart/form-data`, campo `file` (`image/jpeg` ou `image/png`) |
| POST | `/reference?camera_id=cam-01` | Define a imagem abastecida usada como referência visual |
| GET | `/ruptures` | Eventos confirmados (`camera_id`, `only_open`, `include_pending`, `limit`) |
| GET | `/ruptures/export.csv` | Mesmos eventos em CSV (colunas do TCC 2.5-b) |
| POST/GET/DELETE | `/shifts`, `/shifts/{id}` | Escala de funcionários |

Respostas de `/analyze`: **200** (contrato do TCC + campos aditivos), **400** (tipo não suportado, arquivo vazio, imagem
não decodificável, `camera_id` inválido — sempre com `detail`), **413** (acima do limite), **503** (modelo indisponível).
`413` e `503` são adições ao contrato do TCC, que não cobria esses casos.

```bash
curl -X POST "http://localhost:5000/analyze?camera_id=cam-01" -F "file=@teste.jpeg;type=image/jpeg"
```

`/analyze` também retorna `ruptura_visual`. A comparação usa o YOLO para localizar objetos e agrupa padrões pela
posição e aparência; a área é aproximada pela caixa detectada, não por segmentação. Um frame isolado com região vazia
é uma **suspeita**, não uma ruptura temporal confirmada. Nesta etapa, esse resultado visual ainda não é gravado em
`/ruptures`; essa rota lista os eventos mantidos pelo fluxo de persistência existente.

Cadastre a referência antes de analisar as capturas:

```powershell
curl.exe -F "file=@received\cam-01\2026-11-03\221557_887874_0d69cf17172b4ae8a711ccadbb4f701a.jpg;type=image/jpeg" "http://localhost:5000/reference?camera_id=cam-01"
curl.exe -F "file=@received\cam-01\2026-11-03\221802_700732_bd1dd16c77ae4327878c08f5446ecfd4.jpg;type=image/jpeg" "http://localhost:5000/analyze?camera_id=cam-01"
```

Formato da resposta 200 (ilustrativo — os valores dependem do modelo e da imagem):

```json
{
  "status": "ok", "filename": "teste.jpeg", "detections_count": 1, "camera_id": "cam-01", "people_count": 2,
  "detections": [{"class_id": 0, "class_name": "produto_1", "confidence": 0.8731,
                  "bbox": {"x1": 102.4, "y1": 55.0, "x2": 180.9, "y2": 210.3}}],
  "shelf": null,
  "ruptura_visual": {
    "status": "suspeita", "tem_ruptura": true,
    "rupturas": [{"produto": "padrao_026", "area_referencia_px": 3243, "area_ausente_px": 3243,
                  "ocupacao_percentual": 0.0, "ruptura_percentual": 100.0}]
  }
}
```

`shelf` é `null` na comparação visual por histórico. `people_count` é `null` se o modelo de pessoas não estiver
disponível. `ruptura_percentual` é a fração estimada da área de referência ausente; o cálculo usa caixas delimitadoras
do YOLO, portanto é uma aproximação.

Escala (exemplo): `POST /shifts` com
`{"employee":"Ana","aisle":"C10","starts_at":"2026-06-10T07:00:00","ends_at":"2026-06-10T15:00:00"}`.

Imagens: cada captura é guardada com nome único em `received/<camera>/` e `results/<camera>/`, mais `ultima_imagem.*` e
`ultima_deteccao.jpg` por câmera. **Não há rotação automática**: apague imagens antigas conforme a política de retenção
(as imagens podem conter dados de trabalhadores — LGPD, Lei 13.709/2018).

## Treinamento do YOLO11

Você executa o treino; o código abaixo está pronto. O TCC especifica fine-tuning de `yolo11n.pt`, imagem 640 px,
~100 épocas, saída `best.pt`.

1. **Defina as classes.** Edite `names` em `dataset/data.yaml` (provisório: `produto_1..3`). O TCC cita três produtos;
   se forem mais, acrescente linhas. Use os mesmos nomes em `config/shelves.yaml`.
2. **Colete imagens com a própria ESP32-CAM**, na posição, resolução e iluminação do experimento, variando ocupação da
   gôndola, luz e ângulo (TCC 2.5-a). As imagens recebidas pela API ficam em `received/<camera>/` — copie-as para o dataset.
3. **Anote** com uma ferramenta que exporte **YOLO Detection** (ex.: CVAT, Label Studio, Roboflow, LabelImg): um `.txt`
   por imagem, uma linha por objeto: `classe x_centro y_centro largura altura`, valores normalizados de 0 a 1.
   Imagem sem objetos tem `.txt` vazio (fundo). Para o modelo reconhecer *falta* de produto, inclua gôndolas
   vazias/parcialmente ocupadas.
4. **Organize** em `dataset/images/{train,val,test}` e `dataset/labels/{train,val,test}` com os mesmos nomes-base
   (`foto1.jpg` ↔ `foto1.txt`). Uma divisão comum é ~70/20/10, sem repetir a mesma cena entre conjuntos.
5. **Valide a compatibilidade**: `python training/check_dataset.py` (confere pares, formato, intervalo e classes).
6. **Treine**:
   ```bash
   python training/train.py --device 0 --batch 16          # GPU
   python training/train.py --device cpu --batch 8         # CPU (lento)
   python training/train.py --install                      # copia best.pt para models/best.pt
   ```
   Hiperparâmetros: `--epochs 100 --imgsz 640 --patience 30 --seed 0 --workers 4`. Resultados em
   `runs/train/sdp/` (`weights/best.pt`, `results.png`, `confusion_matrix.png`).
7. **Avalie** (precisão, recall, mAP@0.5, mAP@0.5:0.95): `python training/test_model.py --weights models/best.pt --split test`
   — grava `runs/eval/metrics_test.json`. Use `--split val` para a validação.
8. **Integre**: em `.env`, `SDP_MODEL_PATH=models/best.pt` e reinicie. Sem alterar código.

**Como ler os resultados:** *precisão* = das detecções, quantas estavam certas; *recall* = dos produtos reais, quantos
foram achados (recall baixo vira falsa ruptura); *mAP@0.5* = qualidade com sobreposição mínima de 50%; *mAP@0.5:0.95* =
média sobre limiares de 0,50 a 0,95 (mais exigente, sensível à precisão das caixas). Na `confusion_matrix.png`, confusões
entre produtos e “fundo” mostram o que falta no dataset. Após treinar, **reavalie `SDP_CONFIDENCE`**: o limiar passa a
determinar a contagem e, portanto, a ruptura (TCC 6.2).

## Testes

```bash
python -m unittest discover -s tests -t . -v      # sem dependências extras além das do projeto
pytest                                            # equivalente (pip install -r requirements-dev.txt)
```

| Arquivo | Valida | Precisa de |
|---|---|---|
| `test_domain.py` | mínimo por produto (60%), fluxo, ciclo do evento, escala | nada |
| `test_track_rupture.py` | início, confirmação, descarte, fim, responsável | nada |
| `test_analyze_image.py` | validações 400, modelo ausente, gravação, regra de ruptura, pessoas | numpy, opencv |
| `test_report_and_config.py` | relatório, CSV, YAML, modelo sem pesos | pandas, pyyaml |
| `test_check_dataset.py` | validador de dataset | pyyaml |
| `test_sql_repositories.py` | persistência SQLite real | sqlalchemy (pulado se ausente) |
| `test_api.py` | endpoints, 503 sem pesos, contrato 200/400, escala, CSV | fastapi, httpx, sqlalchemy (pulado se ausente) |
| `test_yolo_integration.py` | adaptador com YOLO11 real | ultralytics + `models/yolo11n.pt` (pulado se ausente) |

Nenhum teste exige o `best.pt`. Os dublês substituem só o detector e a persistência em memória, nos testes de regra.

## Docker

```bash
docker compose up --build                       # CPU
# GPU NVIDIA (requer NVIDIA Container Toolkit; SDP_DEVICE=0 no .env):
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Os diretórios `models/`, `config/`, `data/`, `received/` e `results/` são volumes: coloque os pesos em `models/` no
host. Porta 5000. Imagem `python:3.11-slim`, PyTorch CPU por padrão (argumento `TORCH_INDEX_URL` para GPU).

## Firmware (ESP32-CAM)

O firmware recuperado está em `esp32-cam/conf-cam.ino`. Ele envia para `/captures?camera_id=cam-01`, endpoint que
**não existe nesta branch**; a API atual recebe imagens por `/analyze`. Portanto, o firmware ainda não deve ser usado
para validar a ruptura nesta versão. Primeiro valide a sequência de imagens já capturada; depois alinhe o firmware e a
API. Preencha SSID, senha e endereço do servidor localmente antes de compilar; não versione credenciais.

## Problemas comuns

| Sintoma | Causa provável / solução |
|---|---|
| `POST /analyze` retorna 503 | Pesos ausentes: confira `GET /` (`model.error`), `SDP_MODEL_PATH` e rode `scripts/download_weights.py` |
| Câmera sem avaliação de ruptura (`shelf: null`) | `camera_id` sem entrada em `config/shelves.yaml` |
| `unmonitored_classes` preenchido | O modelo carregado não tem aquelas classes (ex.: `yolo11n.pt` com produtos reais) — treine e aponte `SDP_MODEL_PATH` |
| ESP32 não conecta | IP do computador no firmware, porta 5000 liberada no firewall, `--host 0.0.0.0` |
| Inferência lenta | Use GPU (`SDP_DEVICE=0`) ou reduza `SDP_IMGSZ` |
| 400 "camera_id inválido" | Use letras, números, `_` e `-` (até 64 caracteres) |
| Erro de `libGL` em contêiner | Já instalado na imagem; em outras bases instale `libgl1 libglib2.0-0` |

## Verificação: o que foi e o que não foi executado

Ambiente de desenvolvimento desta entrega: Python 3.13 com numpy, OpenCV, pandas, pydantic e PyYAML. O acesso ao PyPI
foi negado (HTTP 403 do proxy), então **fastapi, sqlalchemy, ultralytics, torch e pytest não foram instalados**.

| Item | Situação |
|---|---|
| Domínio, aplicação, relatório/CSV, YAML, validador de dataset, adaptador sem pesos | **Testes executados com sucesso**: 64 testes OK (`python -m unittest`) |
| Sintaxe de todos os arquivos Python | `compileall` OK |
| API FastAPI, repositórios SQLAlchemy, `YoloDetector` com pesos | **Implementados, NÃO executados**: 18 testes escritos, pulados por falta das dependências |
| `train.py`, `test_model.py`, `download_weights.py` | **Implementados, NÃO executados** (exigem ultralytics, pesos e dataset) |
| Dockerfile / compose | **Escritos, NÃO construídos** |
| Treinamento e métricas | **Não realizados** — dependem do seu dataset |

## Limitações e decisões técnicas

- **Fluxo de clientes**: com captura a cada 30 s não é possível identificar a mesma pessoa em capturas seguidas, então o
  sistema usa a **média de pessoas visíveis por captura**, não pessoas únicas. Limites baixo/médio/alto são provisórios.
- **Contagem de pessoas** usa um modelo COCO separado (`SDP_PERSON_MODEL_PATH`), pois o `best.pt` só conhece os produtos.
- **Persistência**: SQLAlchemy com `create_all` (sem Alembic): mudanças futuras de esquema exigem migração manual.
- **Segurança**: HTTP sem criptografia e sem autenticação — aceitável em rede local de laboratório (TCC 8-e), não em
  produção. Há validação de tipo, tamanho e `camera_id` (contra *path traversal*).
- **Concorrência**: serialização por câmera é por processo (um worker).
- A regra de ruptura é “*possível* ruptura”: oclusão, ângulo ou erro do modelo podem reduzir a contagem (TCC 2.7).
- Rastreabilidade requisito → código → teste: [`docs/RASTREABILIDADE.md`](docs/RASTREABILIDADE.md).
