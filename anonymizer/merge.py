"""Объединение результатов NER и regex-слоя перед заменой.

Два правила:
1. regex важнее: если NER-спан попал внутрь найденного формулой (например,
   «8» внутри телефона) — NER-спан выбрасывается, остаётся полный regex-спан.
2. соседние спаны одного типа склеиваются, если между ними только пробелы
   и запятые — так ФИО и составной адрес не рвутся на куски.
"""

from __future__ import annotations

from anonymizer.regex_layer import is_valid_span

SEPARATORS = " ,"


def _overlaps(first: dict, second: dict) -> bool:
    return first["start"] < second["end"] and second["start"] < first["end"]


def merge_entities(text: str, ner: list[dict], regex: list[dict]) -> list[dict]:
    """Возвращает итоговые спаны: regex + непересечённый NER, со склейкой."""
    spans: list[dict] = [dict(entity) for entity in regex]

    for entity in ner:
        # структурные типы проверяем формулой: огрызки вроде «220» из карты
        # или «8» из телефона не маскируем — они портят текст, но не защищают
        if not is_valid_span(entity["type"], entity["text"]):
            continue
        if any(_overlaps(entity, regex_entity) for regex_entity in regex):
            continue
        spans.append(dict(entity))

    spans.sort(key=lambda entity: (entity["start"], entity["end"]))

    merged: list[dict] = []
    for entity in spans:
        if merged and merged[-1]["type"] == entity["type"]:
            gap = text[merged[-1]["end"]: entity["start"]]
            if gap.strip(SEPARATORS) == "":
                merged[-1]["end"] = entity["end"]
                merged[-1]["text"] = text[merged[-1]["start"]: entity["end"]]
                continue
        merged.append(entity)

    return merged
