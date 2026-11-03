# Валидация данных с Great Expectations: набор правил (suite) и его проверка.
# Запуск из корня проекта:  python -m src.data.validation [файл.csv ...]
# Без аргументов проверяет train.csv и test.csv.
# Если хотя бы одна проверка не пройдена — скрипт завершается с ошибкой (код 1).
# Поэтому CI-пайплайн остановится и не будет обучать модель на плохих данных.

import json
import sys
from pathlib import Path
import great_expectations as gx
import pandas as pd

from src.config import ROOT, TRAIN_PATH, TEST_PATH, TARGET, PAY_COLS, BILL_COLS, AMT_COLS

SUITE_NAME = "credit_data_suite"
SUITE_PATH = ROOT / "data" / "expectations" / f"{SUITE_NAME}.json"

FEATURE_COLUMNS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY_COLS + BILL_COLS + AMT_COLS
)
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET]

# Диапазоны допустимых значений
RANGES = {"LIMIT_BAL": (1, 5_000_000), "AGE": (18, 100)}
RANGES.update({c: (-2, 8) for c in PAY_COLS})
RANGES.update({c: (-1_000_000, 5_000_000) for c in BILL_COLS})
RANGES.update({c: (0, 5_000_000) for c in AMT_COLS})

# Допустимые коды категорий (EDUCATION 0/5/6 и MARRIAGE 0 встречаются в данных)
VALUE_SETS = {
    "SEX": [1, 2],
    "EDUCATION": [0, 1, 2, 3, 4, 5, 6],
    "MARRIAGE": [0, 1, 2, 3],
    TARGET: [0, 1],
}


# Создаёт набор правил (Expectation Suite) внутри контекста GE
def make_suite(context):
    suite = context.suites.add(gx.ExpectationSuite(name=SUITE_NAME))
    E = gx.expectations

    # Структура таблицы
    suite.add_expectation(E.ExpectTableColumnsToMatchSet(column_set=ALL_COLUMNS, exact_match=True))
    suite.add_expectation(E.ExpectTableRowCountToBeBetween(min_value=1000))

    # Пропуски и типы — для каждого столбца
    for col in ALL_COLUMNS:
        suite.add_expectation(E.ExpectColumnValuesToNotBeNull(column=col))
        suite.add_expectation(
            E.ExpectColumnValuesToBeInTypeList(column=col, type_list=["int64", "int32", "float64"])
        )

    # Диапазоны
    for col, (lo, hi) in RANGES.items():
        suite.add_expectation(
            E.ExpectColumnValuesToBeBetween(column=col, min_value=lo, max_value=hi)
        )

    # Допустимые значения
    for col, values in VALUE_SETS.items():
        suite.add_expectation(E.ExpectColumnValuesToBeInSet(column=col, value_set=values))

    return suite


# Проверяет таблицу набором правил. Возвращает результат GE (.success = True/False)
def validate(df):
    context = gx.get_context(mode="ephemeral")
    suite = make_suite(context)
    batch_def = (
        context.data_sources.add_pandas("pandas")
        .add_dataframe_asset(name="credit")
        .add_batch_definition_whole_dataframe("batch")
    )
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})
    return batch.validate(suite)


# Список непройденных проверок в читаемом виде
def failed_checks(result):
    return [
        f"{r.expectation_config.type} {r.expectation_config.kwargs.get('column', '')}"
        for r in result.results
        if not r.success
    ]


# Сохраняет набор правил в JSON (data/expectations/)
def save_suite():
    context = gx.get_context(mode="ephemeral")
    suite = make_suite(context)
    SUITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUITE_PATH, "w", encoding="utf-8") as f:
        json.dump(suite.to_json_dict(), f, ensure_ascii=False, indent=2, default=str)


# Можно передать свои файлы, без них проверяются train.csv и test.csv
def main(paths=None):
    save_suite()
    files = [Path(p) for p in paths] if paths else [TRAIN_PATH, TEST_PATH]
    all_ok = True
    for path in files:
        result = validate(pd.read_csv(path))
        n = len(result.results)
        if result.success:
            print(f" OK {path.name}: все {n} проверок пройдены")
        else:
            all_ok = False
            print(f" ERROR {path.name}: не пройдены проверки: {failed_checks(result)}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
