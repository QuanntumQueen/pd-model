"""Загрузка сырых данных, очистка и разбиение на train/test.

Запуск из корня проекта:  python -m src.data.make_dataset
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (RAW_PATH, PROCESSED_DIR, TRAIN_PATH, TEST_PATH,
                        RAW_TARGET, TARGET, TEST_SIZE, RANDOM_STATE)


def load_raw(path=RAW_PATH):
    """Читает исходный xls. Заголовки столбцов — во второй строке файла."""
    return pd.read_excel(path, header=1)


def clean(df):
    """Переименовывает target, удаляет ID и полные дубликаты (найдены в EDA)."""
    df = df.rename(columns={RAW_TARGET: TARGET})
    df = df.drop(columns=["ID"])
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def split(df):
    """Стратифицированное разбиение: доля дефолтов одинакова в train и test."""
    return train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                            stratify=df[TARGET])


def main():
    df = load_raw()
    print("Загружено:       ", df.shape)

    df = clean(df)
    print("После очистки:   ", df.shape)

    train, test = split(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print(f"train: {train.shape}, доля дефолтов {train[TARGET].mean():.4f}")
    print(f"test:  {test.shape}, доля дефолтов {test[TARGET].mean():.4f}")


if __name__ == "__main__":
    main()
