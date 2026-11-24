"""Схема анонимайзера: coarse-классы и маппинг исходных меток датасета.

Coarse-уровень выбран осознанно: анонимайзеру нужен тип подмены, а не
«улица это или город». Гранулярные метки источника сохраняются рядом
в поле `label`, чтобы можно было вернуться к деталям без переразметки.
"""

from __future__ import annotations

# Порядок важен: он фиксируется в чекпоинте (id2label) и потом не меняется.
ENTITY_TYPES: tuple[str, ...] = (
    "PER",       # ФИО: имя, фамилия
    "LOC",       # адрес: город, улица, дом, индекс
    "PHONE",     # телефон
    "EMAIL",     # email
    "LOGIN",     # логин, username
    "DOC",       # паспорт, водительское удостоверение
    "GOV_ID",    # ИНН, СНИЛС
    "BANK_ACC",  # банковский счёт, IBAN
    "CARD",      # номер банковской карты
    "DATE",      # дата рождения
    "PWD",       # пароль
)

# Исходные метки датасета wolframko/russian-pii-66k -> coarse-класс.
SOURCE_TO_COARSE: dict[str, str] = {
    "GIVENNAME": "PER",
    "SURNAME": "PER",
    "CITY": "LOC",
    "STREET": "LOC",
    "BUILDINGNUM": "LOC",
    "ZIPCODE": "LOC",
    "TELEPHONENUM": "PHONE",
    "EMAIL": "EMAIL",
    "USERNAME": "LOGIN",
    "IDCARDNUM": "DOC",
    "DRIVERLICENSENUM": "DOC",
    "TAXNUM": "GOV_ID",
    "SOCIALNUM": "GOV_ID",
    "ACCOUNTNUM": "BANK_ACC",
    "CREDITCARDNUMBER": "CARD",
    "DATEOFBIRTH": "DATE",
    "PASSWORD": "PWD",
}

LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(ENTITY_TYPES)}


def to_coarse(source_label: str) -> str:
    """Возвращает coarse-класс или падает — молча терять метки нельзя."""
    try:
        return SOURCE_TO_COARSE[source_label]
    except KeyError as exc:
        raise KeyError(
            f"нет маппинга для метки {source_label!r}; "
            f"добавь её в SOURCE_TO_COARSE или опиши причину отказа"
        ) from exc
