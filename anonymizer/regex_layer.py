"""Детерминированный слой для структурных ПДн: regex плюс контрольные суммы.

Модель путается на email, паспортах и телефонах в незнакомых формулировках,
а эти типы проверяются формулой — поэтому regex-слой имеет приоритет над NER.
"""

from __future__ import annotations

import re
from typing import Callable


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def luhn(number: str) -> bool:
    """Проверка номера карты по алгоритму Луна."""
    digits = [int(char) for char in _digits(number)]
    if not 13 <= len(digits) <= 19:
        return False

    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def inn_checksum(number: str) -> bool:
    """Контрольная сумма ИНН: 10 цифр у организации, 12 у физлица."""
    digits = _digits(number)
    if len(digits) == 10:
        weights = (2, 4, 10, 3, 5, 9, 4, 6, 8)
        control = sum(int(char) * weight for char, weight in zip(digits, weights)) % 11 % 10
        return control == int(digits[9])
    if len(digits) == 12:
        weights_first = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
        weights_second = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
        first = sum(int(c) * w for c, w in zip(digits, weights_first)) % 11 % 10
        second = sum(int(c) * w for c, w in zip(digits, weights_second)) % 11 % 10
        return first == int(digits[10]) and second == int(digits[11])
    return False


def snils_checksum(number: str) -> bool:
    """Контрольное число СНИЛС."""
    digits = _digits(number)
    if len(digits) != 11:
        return False

    base, control = digits[:9], int(digits[9:])
    total = sum(int(char) * (9 - index) for index, char in enumerate(base))
    if total < 100:
        expected = total
    elif total in (100, 101):
        expected = 0
    else:
        expected = total % 101
        if expected == 100:
            expected = 0
    return expected == control


# Порядок важен: более специфичные типы проверяются раньше.
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
RE_ACCOUNT = re.compile(r"(?<![A-ZА-Я0-9])RU\d{18,20}(?!\d)")
RE_PHONE = re.compile(
    r"(?<![\d+])(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)"
)
RE_SNILS = re.compile(r"\d{3}-\d{3}-\d{3}\s\d{2}")
RE_INN12 = re.compile(r"\d{12}")
RE_INN10 = re.compile(r"\d{10}")
RE_CARD = re.compile(r"(?<!\d)\d(?:[ -]?\d){11,18}(?!\d)")
RE_DOC = re.compile(r"\d{2}\s?\d{2}\s?\d{6}")

# --- адресные части ---------------------------------------------------------
# Маска ставится вместе со словом-маркером: иначе в тексте остаётся «д. <LOC>»,
# а по обрывку «28» нельзя понять, что это номер дома.
STREET_PREFIX = (
    r"ул\.|улиц[а-яё]*|пр-т|пр\.|просп[а-яё]*|пер\.|переул[а-яё]*|"
    r"б-р|бульвар[а-яё]*|ш\.|шоссе|наб\.|набережн[а-яё]*|пл\.|площад[а-яё]*|"
    r"проезд[а-яё]*|туп\."
)
STREET_WORD = (
    r"улиц[а-яё]*|просп[а-яё]*|переул[а-яё]*|бульвар[а-яё]*|шоссе|"
    r"набережн[а-яё]*|площад[а-яё]*|проезд[а-яё]*"
)
NAME = r"[А-ЯЁ][\w\-]*(?:[\s\-][А-ЯЁ][\w\-]*)?"

RE_CITY = re.compile(rf"(?<![\wА-Яа-яё])(?:г\.|город)\s*{NAME}")
RE_STREET = re.compile(
    rf"(?<![\wА-Яа-яё])(?:(?:{STREET_PREFIX})\s+{NAME}|{NAME}\s+(?:{STREET_WORD}))"
)
RE_HOUSE = re.compile(
    r"(?<![\wА-Яа-яё])(?:д\.|дом)\s*\d{1,3}[А-Яа-яё]?"
    r"(?:\s*(?:корп\.|к\.|стр\.|строение)\s*\d{1,3})?"
)
RE_FLAT = re.compile(r"(?<![\wА-Яа-яё])(?:кв\.|квартира)\s*\d{1,4}")
RE_INDEX = re.compile(r"(?<![\wА-Яа-яё])индекс\s*\d{6}")

# --- контекстные правила ----------------------------------------------------
# Значение нельзя опознать по форме (пароль «Qwerty123!» выглядит как слово),
# но рядом стоит подсказка. Спан берётся по группе 1 — маскируется только
# значение, слово-подсказка остаётся в тексте.
RE_PWD = re.compile(
    r"(?i)(?:пароль|password|passwd|pwd)(?:[^:=\n]{0,30})[:=]\s*"
    r"(\"[^\"]{1,64}\"|«[^»]{1,64}»|[^\s,;]{4,64})"
)
RE_TOKEN = re.compile(
    r"(?i)(?:токен|token|api[_-]?key|секретный ключ|secret)(?:[^:=\n]{0,30})[:=]\s*"
    r"(\"[^\"]{1,128}\"|«[^»]{1,128}»|[A-Za-zА-Яа-яё0-9_\-\.]{8,128})"
)
RE_CVV = re.compile(r"(?i)\b(?:cvv2?|cvc2?)\b\s*[:=]?\s*(\d{3,4})\b")

CONTEXT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PWD", RE_PWD),
    ("PWD", RE_TOKEN),
    ("PWD", RE_CVV),
)

# «Пароль: нет» — маскировать нечего, это не значение.
STOP_VALUES = {
    "нет", "не", "истёк", "истек", "забыл", "забыли", "требуется",
    "не требуется", "none", "null", "-", "—",
}

# --- ФИО по контексту -------------------------------------------------------
# После подсказки имя берём целиком (до трёх слов), чтобы не зависеть от того,
# увидела ли модель фамилию в непривычной формулировке.
NAME = r"[А-ЯЁ][а-яё\-]+(?:\s+[А-ЯЁ][а-яё\-]+){0,2}"

RE_FIO_LABEL = re.compile(rf"(?i:(?:фио|ф\.\s?и\.\s?о\.))\s*[:=]?\s*({NAME})")
RE_CITIZEN = re.compile(rf"(?i:(?:гражданин|гражданка|гр\.))\s+({NAME})")
RE_CALLED = re.compile(
    rf"(?i:(?:меня зовут|его зовут|её зовут|ее зовут|зовут))\s+({NAME})"
)
RE_ON_NAME = re.compile(rf"(?i:на имя)\s+({NAME})")

# Минимум слов в имени: одно допустимо, потому что подсказка уже сработала.
NAME_RULES: tuple[tuple[re.Pattern[str], int], ...] = (
    (RE_FIO_LABEL, 1),
    (RE_CITIZEN, 1),
    (RE_CALLED, 1),
    (RE_ON_NAME, 1),
)

# «Гражданин России» — не ФИО, хотя формально похоже.
NAME_STOPWORDS = {
    "россии", "россию", "россия", "российской", "федерации", "руси",
    "москвы", "москве", "москву", "москва",
}

RULES: tuple[tuple[str, re.Pattern[str], Callable[[str], bool] | None], ...] = (
    ("EMAIL", RE_EMAIL, None),
    ("BANK_ACC", RE_ACCOUNT, None),
    ("PHONE", RE_PHONE, None),
    ("GOV_ID", RE_SNILS, snils_checksum),
    ("GOV_ID", RE_INN12, inn_checksum),
    ("GOV_ID", RE_INN10, inn_checksum),
    ("CARD", RE_CARD, luhn),
    ("DOC", RE_DOC, None),
    # адресные части проверяются после цифровых типов: «д. 10» не должно
    # достаться правилу паспорта или карты
    ("LOC", RE_CITY, None),
    ("LOC", RE_STREET, None),
    ("LOC", RE_HOUSE, None),
    ("LOC", RE_FLAT, None),
    ("LOC", RE_INDEX, None),
)


def is_valid_span(entity_type: str, value: str) -> bool:
    """Проверяет, может ли NER-спан быть сущностью такого типа.

    Для структурных типов есть формула — если не сходится, спаны модели
    вроде «8» из телефона или «220» из карты не маскируем: они только
    портят текст, не защищая данные.
    """
    value = value.strip()
    if entity_type == "EMAIL":
        return bool(RE_EMAIL.fullmatch(value))
    if entity_type == "BANK_ACC":
        return bool(RE_ACCOUNT.fullmatch(value))
    if entity_type == "PHONE":
        return bool(RE_PHONE.fullmatch(value))
    if entity_type == "CARD":
        return luhn(value)
    if entity_type == "GOV_ID":
        if RE_SNILS.fullmatch(value):
            return snils_checksum(value)
        return inn_checksum(value)
    if entity_type == "DOC":
        return bool(RE_DOC.fullmatch(value))
    if entity_type == "PWD":
        # обрывки вроде «Qw» — не пароль, они только портят текст
        return len(value) >= 4
    return True

# Паспорт без словаря рядом — слишком похоже на случайные цифры.
DOC_CONTEXT = re.compile(r"(паспорт|серия|документ|выдан)", re.IGNORECASE)
DOC_CONTEXT_WINDOW = 40


def find_regex_entities(text: str) -> list[dict]:
    """Находит структурные ПДн; возвращает спаны в общем формате."""
    found: dict[tuple[int, int], dict] = {}

    for entity_type, pattern, validator in RULES:
        for match in pattern.finditer(text):
            if validator and not validator(match.group()):
                continue

            if entity_type == "DOC":
                window_start = max(0, match.start() - DOC_CONTEXT_WINDOW)
                if not DOC_CONTEXT.search(text[window_start:match.start()]):
                    continue

            key = (match.start(), match.end())
            # первый подходящий тип побеждает: правила идут от специфичного к общему
            found.setdefault(key, {
                "type": entity_type,
                "start": match.start(),
                "end": match.end(),
                "text": match.group(),
            })

    for entity_type, pattern in CONTEXT_RULES:
        for match in pattern.finditer(text):
            start, end = match.start(1), match.end(1)
            value = text[start:end]
            # кавычки вокруг значения — часть оформления, в маску не входят
            if value[:1] in "\"'«" and value[-1:] in "\"'»":
                start, end = start + 1, end - 1
                value = text[start:end]
            if not value.strip() or value.strip().lower() in STOP_VALUES:
                continue
            found.setdefault((start, end), {
                "type": entity_type,
                "start": start,
                "end": end,
                "text": value,
            })

    for pattern, min_words in NAME_RULES:
        for match in pattern.finditer(text):
            start, end = match.start(1), match.end(1)
            value = text[start:end]
            words = value.split()
            if len(words) < min_words:
                continue
            if any(word.lower().strip(".,") in NAME_STOPWORDS for word in words):
                continue
            found.setdefault((start, end), {
                "type": "PER",
                "start": start,
                "end": end,
                "text": value,
            })

    return sorted(found.values(), key=lambda item: item["start"])
