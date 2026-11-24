"""Декодирование предсказанных токен-меток в character-span'ы.

Общий модуль для обучения и инференса: одна и та же логика границ,
иначе метрики и реальные предсказания расходятся.
"""

from __future__ import annotations

# Хвостовая пунктуация приклеивается к последнему токену и не относится к сущности.
TRAILING_PUNCT = ".,;:!?"


def spans_from_labels(
    label_ids: list[int],
    offsets: list[tuple[int, int]],
    text: str,
    id2label: dict[int, str],
) -> set[tuple[str, int, int]]:
    """Собирает character-span'ы из последовательности меток."""
    spans: set[tuple[str, int, int]] = set()
    current_type: str | None = None
    current_start = 0
    current_end = 0

    def close(entity_type: str, start: int, end: int) -> None:
        while start < end and text[start].isspace():
            start += 1
        while end > start and (text[end - 1].isspace() or text[end - 1] in TRAILING_PUNCT):
            end -= 1
        if start < end:
            spans.add((entity_type, start, end))

    for label_id, (token_start, token_end) in zip(label_ids, offsets):
        # Спецтокены и паддинг несут offset (0, 0) — они не продолжают спан.
        # -100 приходит из gold-меток (ignore_index) — трактуем как O.
        label = "O" if token_start == token_end or label_id < 0 else id2label[label_id]

        if label.startswith("B-"):
            if current_type:
                close(current_type, current_start, current_end)
            current_type, current_start, current_end = label[2:], token_start, token_end
        elif label.startswith("I-") and current_type == label[2:]:
            current_end = token_end
        else:
            if current_type:
                close(current_type, current_start, current_end)
            current_type = None

    if current_type:
        close(current_type, current_start, current_end)
    return spans
