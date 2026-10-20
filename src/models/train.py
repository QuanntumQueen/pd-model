# Обучение 5 моделей (GridSearchCV) с логированием в MLflow.
# Запуск: python -m src.models.train
# Создаёт: mlflow.db, models/model.pkl, reports/metrics.json, reports/experiments.csv
import json
import time
import skops.io as sio
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from src.config import ROOT, TRAIN_PATH, TEST_PATH, TARGET, RANDOM_STATE
from src.models.metrics import best_threshold, compute_metrics, plot_roc
from src.models.pipeline import create_pipeline

MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT = "pd-model-credit-default"
MODEL_PATH = ROOT / "models" / "model.pkl"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

# Эксперименты: модель + сетка гиперпараметров
MODELS = {
    "logreg": (
        LogisticRegression(max_iter=1000),
        {"model__C": [0.01, 0.1, 1, 10]}),
    "decision_tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"model__max_depth": [4, 6, 8], "model__min_samples_leaf": [50, 200]}),
    "random_forest": (
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        {"model__max_depth": [8, 12], "model__min_samples_leaf": [20, 50]}),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 4]}),
    "hist_gb": (
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31],
         "model__l2_regularization": [0.0, 1.0]}),
}

# Обучает модель с GridSearchCV и логирует её в MLflow.
def train_one(name, model, param_grid, X_train, y_train, X_test, y_test, cv):
    with mlflow.start_run(run_name=name):
        start = time.time()
        search = GridSearchCV(create_pipeline(model), param_grid, scoring="roc_auc",
                              cv=cv, n_jobs=-1, return_train_score=True)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        # Порог выбираем на train (out-of-fold), test не участвует
        oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        threshold = best_threshold(y_train, oof)

        proba = best.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, proba, threshold)
        i = search.best_index_
        row = {
            "model": name,
            "cv_roc_auc": search.best_score_,
            "cv_roc_auc_std": search.cv_results_["std_test_score"][i],
            "train_roc_auc": search.cv_results_["mean_train_score"][i],
            **{f"test_{k}": v for k, v in test_metrics.items() if k != "threshold"},
            "threshold": threshold,
            "fit_time_sec": time.time() - start,
            "best_params": str({k.replace("model__", ""): v for k, v in search.best_params_.items()}),
        }

        mlflow.set_tag("model_type", name)
        mlflow.log_param("param_grid", str(param_grid))
        mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
        mlflow.log_param("threshold", threshold)
        mlflow.log_metrics({k: v for k, v in row.items()
                            if k not in ("model", "threshold", "best_params")})
        roc_path = plot_roc(y_test, proba, FIG_DIR / f"roc_{name}.png", f"ROC-кривая: {name}")
        mlflow.log_artifact(str(roc_path))
        # Свои классы (FeatureEngineer и др.) MLflow просит явно отметить как доверенные
        trusted = sio.get_untrusted_types(data=sio.dumps(best))
        mlflow.sklearn.log_model(best, name="model", input_example=X_train.head(3),
                                 skops_trusted_types=trusted)
    return best, row

def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    train, test = pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows, fitted = [], {}
    for name, (model, grid) in MODELS.items():
        print(f"▶ {name} ...", flush=True)
        best, row = train_one(name, model, grid, X_train, y_train, X_test, y_test, cv)
        rows.append(row)
        fitted[name] = best
        print(f"  CV ROC-AUC {row['cv_roc_auc']:.4f} | test ROC-AUC {row['test_roc_auc']:.4f}"
              f" | F1 {row['test_f1']:.4f} | порог {row['threshold']} | {row['fit_time_sec']:.0f} с",
              flush=True)

    # Лучшая модель — по ROC-AUC на кросс-валидации
    board = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    board.round(4).to_csv(REPORTS_DIR / "experiments.csv", index=False)
    best_row = board.iloc[0]
    joblib.dump(fitted[best_row["model"]], MODEL_PATH)

    metrics = {k: (round(float(v), 4) if isinstance(v, float) else v)
               for k, v in best_row.items() if k != "best_params"}
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Сравнение моделей:")
    print(board[["model", "cv_roc_auc", "test_roc_auc", "test_gini", "test_f1", "threshold"]]
          .round(4).to_string(index=False))
    print(f"Лучшая модель: {best_row['model']} -> {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
