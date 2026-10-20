"""Тесты валидации данных (Great Expectations).

test_real_data_passes провалится, если в данных появятся аномалии.
Остальные тесты специально портят данные и проверяют, что валидация это ловит.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data.validation import validate, failed_checks

SAMPLE_PATH = Path(__file__).parent / "data" / "sample.csv"


@pytest.fixture
def sample():
    """Свежая копия образца данных для каждого теста."""
    return pd.read_csv(SAMPLE_PATH)


def test_real_data_passes(sample):
    result = validate(sample)
    assert result.success, f"Аномалии в данных: {failed_checks(result)}"


def test_fails_on_null(sample):
    sample.loc[0, "LIMIT_BAL"] = None                      # пропуск
    result = validate(sample)
    assert not result.success
    assert "expect_column_values_to_not_be_null LIMIT_BAL" in failed_checks(result)


def test_fails_on_age_out_of_range(sample):
    sample.loc[0, "AGE"] = 150                             # невозможный возраст
    result = validate(sample)
    assert not result.success
    assert "expect_column_values_to_be_between AGE" in failed_checks(result)


def test_fails_on_unknown_category(sample):
    sample.loc[0, "SEX"] = 3                               # неизвестный код пола
    result = validate(sample)
    assert not result.success
    assert "expect_column_values_to_be_in_set SEX" in failed_checks(result)


def test_fails_on_negative_payment(sample):
    sample.loc[0, "PAY_AMT1"] = -500                       # отрицательный платёж
    result = validate(sample)
    assert not result.success
    assert "expect_column_values_to_be_between PAY_AMT1" in failed_checks(result)


def test_fails_on_missing_column(sample):
    result = validate(sample.drop(columns=["PAY_0"]))      # пропал столбец
    assert not result.success
    assert any("expect_table_columns_to_match_set" in c for c in failed_checks(result))
