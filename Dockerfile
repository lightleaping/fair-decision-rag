FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    EMBEDDING_MODEL_PATH=/app/models/embedding

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.13.0 \
    && pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=6 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
