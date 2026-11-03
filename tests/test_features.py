# Тесты Feature Engineering (src/features/build_features.py)
import pytest
from src.config import AMT_COLS, PAY_COLS
from src.features.build_features import (CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                                         FeatureEngineer, add_features)


# Один клиент из образца — его значения будем менять в тестах
def one_client(sample_df):
    return sample_df.head(1).copy()


def test_n_delays_counts_months_with_delay(sample_df):
    row = one_client(sample_df)
    row[PAY_COLS] = [[2, 1, 0, -1, 0, 2]]          # просрочка в 3 месяцах из 6
    assert add_features(row)["N_DELAYS"].iloc[0] == 3


def test_payment_aggregates(sample_df):
    row = one_client(sample_df)
    row[AMT_COLS] = [[0, 2000, 0, 0, 1500, 0]]     # 4 месяца без платежа
    out = add_features(row)
    assert out["N_ZERO_PAY"].iloc[0] == 4
    assert out["PAY_AMT_MEAN"].iloc[0] == pytest.approx(3500 / 6)


def test_utilization(sample_df):
    row = one_client(sample_df)
    row["LIMIT_BAL"] = 50000
    row["BILL_AMT1"] = 45000
    assert add_features(row)["UTIL"].iloc[0] == pytest.approx(0.9)


def test_rare_codes_merged_into_other(sample_df):
    rows = sample_df.head(4).copy()
    rows["EDUCATION"] = [0, 5, 6, 1]
    rows["MARRIAGE"] = [0, 1, 2, 3]
    out = add_features(rows)
    assert list(out["EDUCATION"]) == [4, 4, 4, 1]
    assert list(out["MARRIAGE"]) == [3, 1, 2, 3]


def test_age_bins(sample_df):
    rows = sample_df.head(5).copy()
    rows["AGE"] = [21, 25, 26, 61, 79]
    out = add_features(rows)
    assert list(out["AGE_BIN"]) == ["21-25", "21-25", "26-30", "61+", "61+"]


def test_input_table_not_modified(sample_df):
    before = sample_df.copy()
    add_features(sample_df)
    assert sample_df.equals(before)


def test_all_model_features_created_without_gaps(sample_df):
    out = add_features(sample_df)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert set(features) <= set(out.columns)
    assert not out[features].isna().any().any()


def test_transformer_matches_function(sample_df):
    out = FeatureEngineer().fit(sample_df).transform(sample_df)
    assert out.equals(add_features(sample_df))
