# Тесты загрузки и очистки данных (src/data/make_dataset.py).
import pandas as pd
from src.config import RAW_TARGET, TARGET
from src.data.make_dataset import clean, split


# Тестовая таблица из 4 клиентов. Клиенты 1 и 3 одинаковые, отличается только ID.
def make_raw():
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "LIMIT_BAL": [10000, 20000, 10000, 30000],
            "AGE": [30, 40, 30, 50],
            RAW_TARGET: [0, 1, 0, 0],
        }
    )


def test_clean_removes_id_and_renames_target():
    out = clean(make_raw())
    assert "ID" not in out.columns
    assert TARGET in out.columns
    assert RAW_TARGET not in out.columns


def test_clean_removes_duplicates():
    out = clean(make_raw())
    assert len(out) == 3
    assert not out.duplicated().any()


def test_split_keeps_default_rate(sample_df):
    train, test = split(sample_df)
    assert len(train) + len(test) == len(sample_df)
    assert abs(len(test) / len(sample_df) - 0.2) < 0.01
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.01
