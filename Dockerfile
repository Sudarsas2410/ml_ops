# Model-serving image for the heart-disease API.
# Build:  docker build -t heart-api:1.0.0 .      (run `python -m src.train` first so models/model.joblib exists)
# Run:    docker run --rm -p 8000:8000 heart-api:1.0.0
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/model.joblib \
    LOG_LEVEL=INFO

WORKDIR /app

# Dependencies first so this layer is cached until requirements change.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src ./src
COPY api ./api
COPY models ./models

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER 10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
