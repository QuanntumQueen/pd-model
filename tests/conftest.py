# "Данные, которые используют все тесты
from pathlib import Path
import pandas as pd
import pytest

@pytest.fixture
def sample_df():
# Образец очищенных данных (1500 клиентов) — новая копия для каждого теста.
    return pd.read_csv(Path(__file__).parent / "data" / "sample.csv")
