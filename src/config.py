"""Настройки проекта: пути, имена столбцов, параметры разбиения."""
from pathlib import Path

# Корень проекта: папка pd-model (на 2 уровня выше этого файла)
ROOT = Path(__file__).resolve().parents[1]

# Пути к данным
RAW_PATH = ROOT / "data" / "raw" / "default of credit card clients.xls"
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

# Целевая переменная: исходное длинное имя и короткое, которым пользуемся дальше
RAW_TARGET = "default payment next month"
TARGET = "default"

# Группы столбцов
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

# Разбиение на train/test
TEST_SIZE = 0.2
RANDOM_STATE = 42
