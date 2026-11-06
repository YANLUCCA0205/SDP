# Experimento 01 — coleta histórica por ESP32-CAM

## Objetivo
Capturar uma imagem a cada 30 segundos e manter o original para análise temporal posterior. **Não** interpretar uma captura salva como ruptura detectada.

## API
`POST /captures?camera_id=cam-01`, multipart campo `file` (JPEG ou PNG); resposta HTTP **201** com `capture_id`, `received_at_utc`, `bytes_count` e `processing_status=PENDING`.

O endpoint **não exige pesos YOLO** e não executa inferência. O endpoint anterior `POST /analyze` permanece independente e inalterado. Os arquivos são gravados em `received/<camera_id>/<YYYY-MM-DD>/` e os metadados na tabela SQLite `captures` (gerada via metadata do SQLAlchemy na inicialização). `PENDING` significa aguardando processamento futuro; **não existe worker automático nesta entrega**.

### Teste no Windows
```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_captures.py -q
uvicorn main:app --host 0.0.0.0 --port 5000
curl.exe -X POST "http://127.0.0.1:5000/captures?camera_id=cam-01" -F "file=@testes_reais/foto_1.jpg;type=image/jpeg"
```
O ambiente que executa o teste deve ter espaço disponível em disco, rede protegida e portas liberadas apenas na rede local. Não exponha HTTP sem autenticação na internet.

### Câmera
No firmware, aponte `apiPath` para `/captures?camera_id=cam-01`. A resposta esperada agora é HTTP 201, não 200. Confira que a ESP32-CAM informa esse código no Monitor Serial e que o arquivo aparece em `received/cam-01/`.

### Capacidade
A cada 30 segundos: 2.880 imagens/dia, sem contar falhas. Reserve alguns gigabytes para uma câmera e monitore disco, falhas e períodos sem capturas. Não há retenção automática, fila na ESP32-CAM ou reenvio garantido neste incremento.

## Próximas etapas
1. Integrar os serviços experimentais de `pattern_masks/` sem executar YOLO repetidamente.
2. Criar processamento assíncrono persistente com estados e retentativas.
3. Comparar imagens da mesma prateleira e validar candidatos a ruptura com dados rotulados.
4. Adicionar proteção por chave de câmera, monitoramento e retenção antes de uso contínuo fora da rede de laboratório.

## Inventário inicial de simplificação
| Área | Situação | Decisão |
|---|---|---|
| `app/api`, `app/application`, `app/domain`, `app/infrastructure` | Núcleo ativo | Preservar |
| `pattern_masks/generate.py` | Experimento HSV, carrega YOLO em separado | Refatorar para receber detecções |
| `pattern_masks/occupancy_analysis.py` | Experimento de máscaras e lacunas, reexecuta YOLO | Refatorar para pipeline único |
| `pattern_masks/detect_gaps.py` | Algoritmo experimental por distâncias entre padrões | Validar e integrar após testes |
| `pattern_masks/shelf_analysis.py` | Regiões fixas para imagem de referência | Generalizar por perfil de câmera |
| `training/` | Treinamento e validação de modelo | Preservar como ferramenta offline |
| `tests/` | Testes do domínio e da API | Preservar e ampliar |
| `testes_reais/`, `weights/`, `yolo11n.pt` | Exemplos/pesos versionados | Avaliar política Git LFS e arquivos grandes; **não remover nesta etapa** |

Não houve remoção automática de scripts: o objetivo desta entrega é estabilizar a coleta antes de refatorar o detector.
