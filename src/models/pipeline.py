# Sklearn Pipeline: Feature Engineering - предобработка (Imputer, Scaler, OHE) - модель
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import FeatureEngineer, NUMERIC_FEATURES, CATEGORICAL_FEATURES


# Числовые: пропуски - медиана, масштабирование. Категориальные: пропуски - мода, OHE
def create_preprocessor():
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


# Полный конвейер: принимает сырые данные клиента, возвращает предсказание модели
def create_pipeline(model):
    return Pipeline(
        [
            ("features", FeatureEngineer()),
            ("preprocessor", create_preprocessor()),
            ("model", model),
        ]
    )
