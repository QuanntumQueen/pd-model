# PD-модель: вероятность дефолта по кредитным картам.
Модель оценки вероятности дефолта (Probability of Default) клиентов банка
на датасете [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (UCI).
## Структура проекта

    data/raw/            исходные данные
    data/processed/      очищенные данные, train/test
    models/              обученные модели
    notebooks/           исследования (01_eda.ipynb)
    reports/figures/     графики
    src/data/            загрузка, очистка, валидация
    src/features/        Feature Engineering
    src/models/          обучение и метрики
    src/api/             FastAPI-сервис
    src/monitoring/      мониторинг дрейфа (PSI)
    tests/               тесты pytest

## Статус

- [x] EDA
- [ ] Подготовка данных и Feature Engineering
- [ ] Валидация данных
- [ ] Обучение модели, MLflow
- [ ] DVC
- [ ] Тесты и CI
- [ ] FastAPI + Docker
- [ ] Мониторинг
