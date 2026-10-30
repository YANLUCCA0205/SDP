# Imagem da API do SDP (Python 3.11, como no TCC).
#
# Build padrão = PyTorch somente CPU (imagem bem menor):
#   docker build -t sdp .
# Build para GPU NVIDIA (wheels CUDA do PyPI):
#   docker build --build-arg TORCH_INDEX_URL=https://pypi.org/simple -t sdp:gpu .
FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Bibliotecas de sistema exigidas pelo OpenCV (o ultralytics instala opencv-python).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --index-url ${TORCH_INDEX_URL} --extra-index-url https://pypi.org/simple torch torchvision \
    && pip install -r requirements.txt

COPY . .

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/', timeout=3)" || exit 1

# Um único worker: a serialização por câmera da regra de ruptura é por processo.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
