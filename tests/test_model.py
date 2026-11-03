# Тесты модели: Pipeline и метрики (src/models/pipeline.py, src/models/metrics.py)
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import TARGET
from src.models.metrics import best_threshold, compute_metrics
from src.models.pipeline import create_pipeline


# Pipeline с логистической регрессией, обученный на образце
@pytest.fixture
def fitted(sample_df):
    X, y = sample_df.drop(columns=[TARGET]), sample_df[TARGET]
    model = create_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    return model, X, y


def test_pipeline_returns_probabilities(fitted):
    model, X, _ = fitted
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert set(model.predict(X)) <= {0, 1}


def test_pipeline_handles_missing_values(fitted):
    model, X, _ = fitted
    X_bad = X.head(5).copy()
    X_bad.loc[X_bad.index[0], "LIMIT_BAL"] = np.nan  # неполная заявка
    X_bad.loc[X_bad.index[1], "AGE"] = np.nan
    proba = model.predict_proba(X_bad)[:, 1]
    assert not np.isnan(proba).any()


def test_model_better_than_random(fitted):
    model, X, y = fitted
    assert roc_auc_score(y, model.predict_proba(X)[:, 1]) > 0.65


def test_compute_metrics_on_perfect_model():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, proba, threshold=0.5)
    assert m["roc_auc"] == 1.0
    assert m["gini"] == 1.0
    assert m["f1"] == 1.0


def test_best_threshold_in_range():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 200)
    proba = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    assert 0.05 <= best_threshold(y, proba) <= 0.95
