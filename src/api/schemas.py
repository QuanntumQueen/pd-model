# Схемы запроса и ответа API (проверка входных данных через Pydantic)
from pydantic import BaseModel, ConfigDict, Field

EXAMPLE = {
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 46000,
    "BILL_AMT2": 45000,
    "BILL_AMT3": 44000,
    "BILL_AMT4": 40000,
    "BILL_AMT5": 38000,
    "BILL_AMT6": 36000,
    "PAY_AMT1": 0,
    "PAY_AMT2": 1500,
    "PAY_AMT3": 1500,
    "PAY_AMT4": 1500,
    "PAY_AMT5": 1500,
    "PAY_AMT6": 1500,
}


# Данные клиента за последние 6 месяцев (как в исходном датасете)
class Client(BaseModel):

    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE})

    LIMIT_BAL: float = Field(gt=0, description="Кредитный лимит")
    SEX: int = Field(ge=1, le=2, description="Пол: 1 — мужчина, 2 — женщина")
    EDUCATION: int = Field(ge=0, le=6, description="Образование (код)")
    MARRIAGE: int = Field(ge=0, le=3, description="Семейное положение (код)")
    AGE: int = Field(ge=18, le=100, description="Возраст")
    PAY_0: int = Field(ge=-2, le=8, description="Статус платежа, сентябрь")
    PAY_2: int = Field(ge=-2, le=8, description="Статус платежа, август")
    PAY_3: int = Field(ge=-2, le=8, description="Статус платежа, июль")
    PAY_4: int = Field(ge=-2, le=8, description="Статус платежа, июнь")
    PAY_5: int = Field(ge=-2, le=8, description="Статус платежа, май")
    PAY_6: int = Field(ge=-2, le=8, description="Статус платежа, апрель")
    BILL_AMT1: float = Field(description="Сумма долга, сентябрь")
    BILL_AMT2: float = Field(description="Сумма долга, август")
    BILL_AMT3: float = Field(description="Сумма долга, июль")
    BILL_AMT4: float = Field(description="Сумма долга, июнь")
    BILL_AMT5: float = Field(description="Сумма долга, май")
    BILL_AMT6: float = Field(description="Сумма долга, апрель")
    PAY_AMT1: float = Field(ge=0, description="Сумма платежа, сентябрь")
    PAY_AMT2: float = Field(ge=0, description="Сумма платежа, август")
    PAY_AMT3: float = Field(ge=0, description="Сумма платежа, июль")
    PAY_AMT4: float = Field(ge=0, description="Сумма платежа, июнь")
    PAY_AMT5: float = Field(ge=0, description="Сумма платежа, май")
    PAY_AMT6: float = Field(ge=0, description="Сумма платежа, апрель")


# Ответ модели
class Prediction(BaseModel):

    default_class: int = Field(description="1 — ожидается дефолт, 0 — нет")
    default_probability: float = Field(description="Вероятность дефолта (PD)")
    threshold: float = Field(description="Порог: при вероятности не ниже порога класс = 1")
