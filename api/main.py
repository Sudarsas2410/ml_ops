"""FastAPI model-serving application.

Endpoints
---------
POST /predict     JSON patient record -> prediction, label, confidence
GET  /health      liveness probe (process is up)
GET  /ready       readiness probe (model loaded)
GET  /model-info  metadata about the served model
GET  /metrics     Prometheus metrics
GET  /docs        interactive OpenAPI docs

Run locally:  uvicorn api.main:app --reload
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import __version__, config
from src.predict import load_model, predict

logger = logging.getLogger("heart_api")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class JsonFormatter(logging.Formatter):
    """One JSON object per line: easy to read, grep and ship to a log stack."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_file = os.getenv("LOG_FILE")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    for handler in handlers:
        handler.setFormatter(JsonFormatter())
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = False


def log_event(message: str, **fields) -> None:
    logger.info(message, extra={"extra_fields": fields})


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class PatientFeatures(BaseModel):
    """The 13 clinical features used by the model."""

    model_config = ConfigDict(json_schema_extra={"examples": [config.SAMPLE_PATIENT]})

    age: float = Field(..., ge=1, le=120, description=config.FEATURE_DESCRIPTIONS["age"])
    sex: int = Field(..., description=config.FEATURE_DESCRIPTIONS["sex"])
    cp: int = Field(..., description=config.FEATURE_DESCRIPTIONS["cp"])
    trestbps: float = Field(..., ge=50, le=260, description=config.FEATURE_DESCRIPTIONS["trestbps"])
    chol: float = Field(..., ge=80, le=700, description=config.FEATURE_DESCRIPTIONS["chol"])
    fbs: int = Field(..., description=config.FEATURE_DESCRIPTIONS["fbs"])
    restecg: int = Field(..., description=config.FEATURE_DESCRIPTIONS["restecg"])
    thalach: float = Field(..., ge=50, le=250, description=config.FEATURE_DESCRIPTIONS["thalach"])
    exang: int = Field(..., description=config.FEATURE_DESCRIPTIONS["exang"])
    oldpeak: float = Field(..., ge=0, le=10, description=config.FEATURE_DESCRIPTIONS["oldpeak"])
    slope: int = Field(..., description=config.FEATURE_DESCRIPTIONS["slope"])
    ca: int = Field(..., description=config.FEATURE_DESCRIPTIONS["ca"])
    thal: int = Field(..., description=config.FEATURE_DESCRIPTIONS["thal"])

    @field_validator(*config.VALID_CATEGORIES)
    @classmethod
    def _allowed_category(cls, value: int, info) -> int:
        allowed = config.VALID_CATEGORIES[info.field_name]
        if value not in allowed:
            raise ValueError(f"must be one of {sorted(allowed)}")
        return value


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease predicted, 0 = not predicted")
    label: str
    confidence: float = Field(..., ge=0, le=1, description="Probability of the predicted class")
    probability_disease: float = Field(..., ge=0, le=1)
    model_version: str


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #
def create_app(model_path: Path | str | None = None) -> FastAPI:
    """Build an app with its own metrics registry (keeps tests isolated)."""
    configure_logging()
    model_path = Path(model_path or os.getenv("MODEL_PATH", config.MODEL_FILE))
    metrics_file = Path(os.getenv("METRICS_PATH", model_path.parent / "metrics.json"))

    registry = CollectorRegistry()
    ProcessCollector(registry=registry)
    http_requests = Counter(
        "http_requests_total", "HTTP requests", ["method", "handler", "status"], registry=registry
    )
    http_latency = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "handler"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
        registry=registry,
    )
    in_progress = Gauge("http_requests_in_progress", "Requests currently being served", registry=registry)
    predictions = Counter("heart_predictions_total", "Predictions served", ["prediction"], registry=registry)
    confidence_hist = Histogram(
        "heart_prediction_confidence",
        "Confidence of served predictions",
        buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
        registry=registry,
    )
    model_info = Gauge("heart_model_info", "Served model metadata", ["version", "model"], registry=registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        app.state.meta = {}
        try:
            app.state.model = load_model(model_path)
            if metrics_file.exists():
                app.state.meta = json.loads(metrics_file.read_text())
            name = app.state.meta.get("best_model", "unknown")
            model_info.labels(version=app.state.meta.get("version", __version__), model=name).set(1)
            log_event("model loaded", model_path=str(model_path), model=name)
        except FileNotFoundError:
            logger.error("model file not found at %s; /ready will report 503", model_path)
        yield
        log_event("shutting down")

    app = FastAPI(
        title="Heart Disease Risk API",
        description="Predicts the presence of heart disease from 13 clinical features "
        "(UCI Cleveland dataset). Educational project, not a medical device.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        started = time.perf_counter()
        in_progress.inc()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            in_progress.dec()
            elapsed = time.perf_counter() - started
            route = request.scope.get("route")
            handler = getattr(route, "path", "unmatched")  # bounded label cardinality
            http_requests.labels(request.method, handler, str(status)).inc()
            http_latency.labels(request.method, handler).observe(elapsed)
            if handler != "/metrics":
                log_event(
                    "request",
                    request_id=request_id,
                    method=request.method,
                    path=request.url.path,
                    status=status,
                    latency_ms=round(elapsed * 1000, 2),
                    client=request.client.host if request.client else None,
                )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/", include_in_schema=False)
    def root():
        return {"service": "heart-disease-api", "docs": "/docs", "predict": "POST /predict"}

    @app.get("/health", tags=["ops"])
    def health():
        """Liveness: the process is running."""
        return {"status": "ok", "version": __version__}

    @app.get("/ready", tags=["ops"])
    def ready(request: Request):
        """Readiness: the model is loaded and the service can take traffic."""
        if request.app.state.model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"status": "ready"}

    @app.get("/model-info", tags=["ops"])
    def model_info_endpoint(request: Request):
        if request.app.state.model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        meta = request.app.state.meta
        return {
            "model": meta.get("best_model", "unknown"),
            "version": meta.get("version", __version__),
            "trained_at": meta.get("trained_at"),
            "features": config.FEATURES,
            "cv_roc_auc": meta.get("cv", {}).get("roc_auc"),
            "test_roc_auc": meta.get("test", {}).get("roc_auc"),
        }

    @app.post("/predict", response_model=PredictionResponse, tags=["inference"])
    def predict_endpoint(patient: PatientFeatures, request: Request):
        """Score one patient record and return the prediction with its confidence."""
        model = request.app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        result = predict(model, patient.model_dump())[0]
        predictions.labels(str(result["prediction"])).inc()
        confidence_hist.observe(result["confidence"])
        log_event("prediction", prediction=result["prediction"], confidence=result["confidence"])
        return PredictionResponse(**result, model_version=request.app.state.meta.get("version", __version__))

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
