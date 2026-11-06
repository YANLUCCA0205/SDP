# SDP — Capturas reais e treinamento YOLO11

## Estado dos dados
As 13 capturas de `received/cam-01/2026-11-03` sao imagens reais, **nao um dataset supervisionado**: ainda nao ha caixas YOLO verificadas. As mascaras de diferenca temporal nao equivalem a rotulos de produtos.

## 1. Preparar sem alterar as fotos
Na raiz do SDP:

```powershell
python training/prepare_captures.py --camera-id cam-01 --day 2026-11-03
```

O resultado fica em `dataset/to_annotate/cam-01/2026-11-03/images/` e `manifest.csv`. O script checa integridade, calcula SHA256, ignora duplicatas identicas e **nao cria arquivos de rotulo vazios**. Essa pasta e ignorada pelo Git.

## 2. Anotar
Usar CVAT, Label Studio ou Roboflow com exportacao **YOLO Detection**. Marcar **cada embalagem visivel** com caixa delimitadora. Uma classe generica `object` ensina a localizar embalagens, mas **nao** diferencia OMO, Assim ou Brilhante. Se for preciso reconhecer esses padroes, criar classes especificas e revisar exemplos de cada uma. Espacos vazios nao sao rotulos de produtos. Conferir visualmente as anotacoes; nao converter mascaras temporais diretamente em verdade de treinamento.

## 3. Separar treino, validacao e teste
As 13 fotos sao de **uma unica cena em sequencia**. Nao distribuir fotos quase identicas aleatoriamente entre `train`, `val` e `test`: isso superestima a acuracia. Coletar **sessoes/cenas adicionais**, variar angulo, luz, ocupacao e rupturas, e separar por sessao. Reservar 03/11 como sequencia temporal de demonstracao, se possivel.

Apos exportacao e revisao das anotacoes, preencher `dataset/images/{train,val,test}` e `dataset/labels/{train,val,test}` com os pares de nomes correspondentes, e ajustar `dataset/data.yaml` com as classes reais. Conferir coerencia com `config/shelves.yaml`.

## 4. Validar e treinar
```powershell
python training/check_dataset.py
python training/train.py --device 0 --batch 4 --workers 0 --epochs 100 --install
python training/test_model.py --weights models/best.pt --split test --device 0 --batch 4
```

`train.py` agora recusa estrutura de dataset invalida e treinamento sem nenhuma instancia anotada. Para RTX 3050 4 GB, diminuir `--batch` se ocorrer falta de memoria. **Nao ha best.pt treinado nem metricas reais ate executar o treino com anotacoes.**

Depois, configurar `SDP_MODEL_PATH=models/best.pt` em `.env` e reiniciar a API. Revisar a confianca e validar as rupturas contra as fotografias originais.
