import json

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src import config


@pytest.fixture()
def client(model_file):
    with TestClient(create_app(model_file)) as c:
        yield c


@pytest.fixture()
def no_model_client(tmp_path):
    with TestClient(create_app(tmp_path / "absent.joblib")) as c:
        yield c


def payload(**overrides):
    return {**config.SAMPLE_PATIENT, **overrides}


# ------------------------------- happy path ------------------------------ #
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ready_when_model_loaded(client):
    assert client.get("/ready").json() == {"status": "ready"}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model"] == "logistic_regression"
    assert body["features"] == config.FEATURES


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=payload())
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert body["label"] in {"heart disease", "no heart disease"}
    assert body["model_version"] == "test"


def test_predict_is_deterministic(client):
    assert client.post("/predict", json=payload()).json() == client.post("/predict", json=payload()).json()


def test_predict_accepts_integer_valued_floats(client):
    assert client.post("/predict", json=payload(sex=1.0, cp=4.0)).status_code == 200


def test_higher_risk_profile_scores_higher(client):
    low = payload(age=35, sex=0, cp=3, thalach=185, exang=0, oldpeak=0.0, ca=0, thal=3)
    high = payload(age=66, sex=1, cp=4, thalach=105, exang=1, oldpeak=4.0, ca=3, thal=7)
    p_low = client.post("/predict", json=low).json()["probability_disease"]
    p_high = client.post("/predict", json=high).json()["probability_disease"]
    assert p_high > p_low


# ------------------------------ validation ------------------------------ #
@pytest.mark.parametrize(
    "overrides",
    [
        {"age": 500},
        {"age": -3},
        {"chol": 5},
        {"trestbps": 1000},
        {"oldpeak": -1},
        {"sex": 2},
        {"cp": 9},
        {"thal": 5},
        {"ca": 4},
        {"age": "old"},
    ],
)
def test_invalid_values_rejected_with_422(client, overrides):
    assert client.post("/predict", json=payload(**overrides)).status_code == 422


def test_missing_field_rejected(client):
    body = payload()
    body.pop("chol")
    r = client.post("/predict", json=body)
    assert r.status_code == 422
    assert "chol" in json.dumps(r.json())


def test_empty_and_malformed_bodies_rejected(client):
    assert client.post("/predict", json={}).status_code == 422
    assert (
        client.post("/predict", content="not json", headers={"content-type": "application/json"}).status_code
        == 422
    )


def test_wrong_method_and_unknown_route(client):
    assert client.get("/predict").status_code == 405
    assert client.get("/nope").status_code == 404


# ------------------------------ degraded mode ---------------------------- #
def test_service_without_model_is_alive_but_not_ready(no_model_client):
    assert no_model_client.get("/health").status_code == 200
    assert no_model_client.get("/ready").status_code == 503
    assert no_model_client.post("/predict", json=payload()).status_code == 503


# ------------------------- monitoring and logging ------------------------ #
def test_request_id_header_is_set_and_echoed(client):
    assert client.get("/health").headers["x-request-id"]
    assert client.get("/health", headers={"x-request-id": "abc123"}).headers["x-request-id"] == "abc123"


def test_metrics_endpoint_reflects_traffic(client):
    client.post("/predict", json=payload())
    client.post("/predict", json=payload(age=500))  # 422
    text = client.get("/metrics").text
    assert 'http_requests_total{handler="/predict",method="POST",status="200"} 1.0' in text
    assert 'http_requests_total{handler="/predict",method="POST",status="422"} 1.0' in text
    assert "http_request_duration_seconds_bucket" in text
    assert "heart_predictions_total" in text
    assert "heart_prediction_confidence_count 1.0" in text
    assert 'heart_model_info{model="logistic_regression",version="test"} 1.0' in text


def test_unknown_paths_do_not_explode_metric_cardinality(client):
    for i in range(5):
        client.get(f"/random-{i}")
    text = client.get("/metrics").text
    assert 'handler="unmatched"' in text and "random-0" not in text


def test_requests_are_logged_as_json_lines(model_file, tmp_path, monkeypatch):
    log_file = tmp_path / "logs" / "api.log"
    monkeypatch.setenv("LOG_FILE", str(log_file))
    with TestClient(create_app(model_file)) as c:
        c.post("/predict", json=payload())
    records = [json.loads(line) for line in log_file.read_text().splitlines()]
    request_logs = [r for r in records if r["message"] == "request" and r["path"] == "/predict"]
    assert request_logs and request_logs[0]["status"] == 200 and request_logs[0]["latency_ms"] >= 0
    assert any(r["message"] == "prediction" for r in records)
