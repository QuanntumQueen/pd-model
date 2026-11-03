# REST API модели вероятности дефолта. Запуск: uvicorn src.api.app:app --port 8000
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from src.api.schemas import Client, Prediction
from src.config import ROOT

MODEL_PATH = ROOT / "models" / "model.pkl"
METRICS_PATH = ROOT / "reports" / "metrics.json"

app = FastAPI(title="PD-модель: вероятность дефолта по кредитной карте", version="1.0")
state = {}


# Загружает модель и порог один раз, при первом запросе
def load_model():
    if "model" not in state:
        state["model"] = joblib.load(MODEL_PATH)
        with open(METRICS_PATH, encoding="utf-8") as f:
            state["threshold"] = json.load(f)["threshold"]
    return state["model"], state["threshold"]


# Проверка, что сервис работает
@app.get("/")
def health():
    return {"status": "ok"}


# Принимает данные клиента, возвращает класс и вероятность дефолта
@app.post("/predict", response_model=Prediction)
def predict(client: Client):
    model, threshold = load_model()
    X = pd.DataFrame([client.model_dump()])
    proba = float(model.predict_proba(X)[0, 1])
    return Prediction(
        default_class=int(proba >= threshold),
        default_probability=round(proba, 4),
        threshold=threshold,
    )
