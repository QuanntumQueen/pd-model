"""Feature Engineering. Обоснование каждого признака — в notebooks/01_eda.ipynb."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import PAY_COLS, BILL_COLS, AMT_COLS

# Биннинг возраста (EDA, шаг 22). Верхняя граница 120 — запас для новых клиентов старше 79
AGE_BINS = [20, 25, 30, 35, 40, 50, 60, 120]
AGE_LABELS = ["21-25", "26-30", "31-35", "36-40", "41-50", "51-60", "61+"]

# Списки признаков для модели
NEW_NUMERIC = ["N_DELAYS", "N_ZERO_PAY", "PAY_AMT_MEAN", "UTIL", "UTIL_MEAN"]
NUMERIC_FEATURES = ["LIMIT_BAL", "AGE"] + PAY_COLS + BILL_COLS + AMT_COLS + NEW_NUMERIC
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE", "AGE_BIN"]


def add_features(df):
    """Возвращает копию таблицы с очищенными категориями и новыми признаками."""
    df = df.copy()

    # Недокументированные коды -> «другое» (EDA, шаг 15)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # Биннинг возраста: U-образная зависимость риска
    df["AGE_BIN"] = pd.cut(df["AGE"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)

    # Агрегаты платежей за 6 месяцев
    df["N_DELAYS"] = (df[PAY_COLS] > 0).sum(axis=1)       # месяцев с просрочкой
    df["N_ZERO_PAY"] = (df[AMT_COLS] == 0).sum(axis=1)    # месяцев с нулевым платежом
    df["PAY_AMT_MEAN"] = df[AMT_COLS].mean(axis=1)        # средний платёж

    # Утилизация лимита
    df["UTIL"] = df["BILL_AMT1"] / df["LIMIT_BAL"]
    df["UTIL_MEAN"] = df[BILL_COLS].mean(axis=1) / df["LIMIT_BAL"]
    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Обёртка над add_features для sklearn Pipeline.

    Признаки создаются внутри модели, поэтому API передаёт ей сырые данные клиента,
    и при обучении и в работе считаются одни и те же формулы.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_features(X)
