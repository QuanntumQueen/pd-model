"""Метрики качества, выбор порога классификации и ROC-кривая."""
import matplotlib
matplotlib.use("Agg")  # рисуем в файл, без экрана (нужно для CI и Docker)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def compute_metrics(y_true, proba, threshold=0.5):
    """ROC-AUC, Gini — по вероятностям; Precision, Recall, F1 — по классам при пороге."""
    pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba)
    return {
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "threshold": threshold,
    }


def best_threshold(y_true, proba):
    """Порог, при котором F1 максимален (перебор от 0.05 до 0.95)."""
    thresholds = np.arange(0.05, 0.951, 0.01)
    scores = [f1_score(y_true, (proba >= t).astype(int)) for t in thresholds]
    return round(float(thresholds[int(np.argmax(scores))]), 2)


def plot_roc(y_true, proba, path, title="ROC-кривая"):
    """Строит ROC-кривую и сохраняет её в файл."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"модель (ROC-AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="случайное угадывание")
    ax.set_xlabel("False Positive Rate (доля ложных тревог)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
