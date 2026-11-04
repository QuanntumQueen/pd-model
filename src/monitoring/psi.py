# Мониторинг дрейфа данных: PSI (Population Stability Index)
# Запуск: python -m src.monitoring.psi
# Сравнивает распределения признаков и прогноза модели на train (эталон)
# и на новых данных: test и test с искусственным дрейфом

import joblib
import numpy as np
import pandas as pd

from src.config import ROOT, TARGET, TEST_PATH, TRAIN_PATH
from src.features.build_features import add_features

MODEL_PATH = ROOT / "models" / "model.pkl"
REPORT_PATH = ROOT / "reports" / "psi_report.csv"

# Признаки для мониторинга: самые важные по EDA и новые агрегаты
FEATURES = ["LIMIT_BAL", "AGE", "PAY_0", "BILL_AMT1", "PAY_AMT1", "N_DELAYS", "N_ZERO_PAY", "UTIL"]

# Пороги PSI, принятые в кредитном скоринге
STABLE, MODERATE = 0.1, 0.25
EPS = 1e-4  # чтобы не делить на ноль в пустых интервалах


# Границы интервалов по эталонным данным: квантили или, для дискретных признаков, сами значения
def bin_edges(expected, bins=10):
    values = np.unique(expected)
    if len(values) <= bins:
        cuts = (values[:-1] + values[1:]) / 2
    else:
        cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.concatenate([[-np.inf], cuts, [np.inf]])


# PSI = сумма (доля_новых - доля_эталона) * ln(доля_новых / доля_эталона) по интервалам
def psi(expected, actual, bins=10):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = bin_edges(expected, bins)
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value):
    if value < STABLE:
        return "стабильно"
    if value < MODERATE:
        return "умеренный сдвиг"
    return "значительный сдвиг"


# Таблица PSI по признакам и по прогнозу модели (если модель передана)
def psi_report(reference, current, model=None):
    ref, cur = add_features(reference), add_features(current)
    rows = [{"feature": col, "psi": psi(ref[col], cur[col])} for col in FEATURES]
    if model is not None:
        ref_score = model.predict_proba(reference)[:, 1]
        cur_score = model.predict_proba(current)[:, 1]
        rows.append({"feature": "score (PD)", "psi": psi(ref_score, cur_score)})
    report = pd.DataFrame(rows)
    report["psi"] = report["psi"].round(4)
    report["status"] = report["psi"].map(psi_status)
    return report


# Искусственный дрейф: лимиты снижены на 30%, у всех клиентов просрочка на месяц больше
def simulate_drift(df):
    df = df.copy()
    df["LIMIT_BAL"] = df["LIMIT_BAL"] * 0.7
    df["PAY_0"] = (df["PAY_0"] + 1).clip(upper=8)
    return df


def main():
    model = joblib.load(MODEL_PATH)
    train = pd.read_csv(TRAIN_PATH).drop(columns=[TARGET])
    test = pd.read_csv(TEST_PATH).drop(columns=[TARGET])

    reports = []
    for name, current in [("test", test), ("drift", simulate_drift(test))]:
        report = psi_report(train, current, model)
        report.insert(0, "scenario", name)
        reports.append(report)
        print(f"\nPSI: train -> {name}")
        print(report.drop(columns="scenario").to_string(index=False))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(reports).to_csv(REPORT_PATH, index=False)
    print(f"\nОтчёт сохранён: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
