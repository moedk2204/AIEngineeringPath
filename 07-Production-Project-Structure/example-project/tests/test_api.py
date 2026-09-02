"""API tests through a real HTTP client - the contract, not the internals."""
from sample_data import VALID_PATIENT


def test_health_reports_model_state(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert isinstance(body["model_loaded"], bool)


def test_predict_returns_a_probability(client):
    r = client.post("/predict", json=VALID_PATIENT)
    if r.status_code == 503:
        return                       # no model artifact - run scripts/train.py first
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["flagged"], bool)
    assert body["model_version"]


def test_invalid_payload_is_422_not_500(client):
    r = client.post("/predict", json={**VALID_PATIENT, "age": 999})
    assert r.status_code == 422       # the whole point of the schema layer


def test_batch_respects_its_limit(client):
    r = client.post("/predict/batch", json={"patients": []})
    assert r.status_code == 422       # min_length=1
