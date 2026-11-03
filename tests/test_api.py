# Тесты REST API (src/api/app.py)
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import src.api.app as api
from src.api.schemas import EXAMPLE
from src.config import TARGET
from src.models.pipeline import create_pipeline


# Клиент API с небольшой моделью, обученной на образце данных
@pytest.fixture
def client(sample_df):
    X, y = sample_df.drop(columns=[TARGET]), sample_df[TARGET]
    api.state["model"] = create_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    api.state["threshold"] = 0.5
    yield TestClient(api.app)
    api.state.clear()


def test_health(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_class_and_probability(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["default_class"] in (0, 1)
    assert 0 <= body["default_probability"] <= 1
    assert body["threshold"] == 0.5


def test_predict_rejects_wrong_value(client):
    r = client.post("/predict", json=dict(EXAMPLE, SEX=3))
    assert r.status_code == 422


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in EXAMPLE.items() if k != "AGE"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
