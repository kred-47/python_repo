"""Сверка анонимайзера с тест-набором: что закрыто, что осталось, что закрыто лишнего.

Запуск:
    python -m anonymizer.check
    python -m anonymizer.check --only 3

Вывод устроен так, чтобы итог был виден сразу: сначала исходный текст, ниже —
результат маскирования, затем построчная сверка ожиданий, где у каждого
«частично» перечислены закрытые и оставшиеся открытыми слова.
Код возврата: 0 — провалов нет, 1 — есть.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from anonymizer.cases import CASES
from anonymizer.predict import DEFAULT_MODEL, Detector, anonymize

PARTIAL_THRESHOLD = 0.999
WRAP = 92

LEGEND = """\
Как читать:
  найдено              — ожидаемое значение замаскировано целиком;
  частично N%          — закрыта только часть значения, N — доля закрытых символов;
  пропуск              — не закрыто ничего, это провал;
  необязательно: ...   — такие данные искать не обязательно, на итог не влияет."""


def _covered_ranges(start: int, end: int, spans: list[dict],
                    entity_type: str | None) -> list[list[int]]:
    """Слитые интервалы диапазона, закрытые спанами нужного типа."""
    intervals = sorted(
        (max(start, span["start"]), min(end, span["end"]))
        for span in spans
        if (entity_type is None or span["type"] == entity_type)
        and span["start"] < end
        and span["end"] > start
    )
    merged: list[list[int]] = []
    for begin, finish in intervals:
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], finish)
        else:
            merged.append([begin, finish])
    return merged


def _coverage(start: int, end: int, spans: list[dict], entity_type: str | None) -> float:
    """Доля символов диапазона, закрытая спанами нужного типа."""
    closed = sum(finish - begin for begin, finish in _covered_ranges(start, end, spans, entity_type))
    return closed / max(1, end - start)


def _split_value(text: str, start: int, end: int, spans: list[dict],
                 entity_type: str) -> tuple[list[str], list[str]]:
    """Делит значение на закрытые маской и оставшиеся открытыми куски."""
    closed: list[str] = []
    opened: list[str] = []
    cursor = start
    for begin, finish in _covered_ranges(start, end, spans, entity_type):
        if begin > cursor:
            opened.append(text[cursor:begin])
        closed.append(text[begin:finish])
        cursor = finish
    if cursor < end:
        opened.append(text[cursor:end])
    return (
        [part.strip() for part in closed if part.strip()],
        [part.strip() for part in opened if part.strip()],
    )


def _quote(parts: list[str], limit: int = 60) -> str:
    return ", ".join(f"«{part if len(part) <= limit else part[:limit - 1] + '…'}»" for part in parts)


def _block(title: str, text: str) -> None:
    print(f"\n{title}:")
    lines = textwrap.wrap(
        text,
        width=WRAP,
        initial_indent="  ",
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=False,
    ) or ["  (пустой текст)"]
    for line in lines:
        print(line)


def check_case(detector: Detector, case: dict) -> dict:
    text = case["text"]
    spans = detector.detect(text)

    expected_ranges: list[tuple[int, int]] = []
    results = []

    for kind in ("required", "optional"):
        for entity_type, value in case[kind]:
            start = text.find(value)
            if start < 0:
                results.append({
                    "kind": kind,
                    "type": entity_type,
                    "value": value,
                    "coverage": None,
                    "closed": [],
                    "opened": [],
                })
                continue
            end = start + len(value)
            expected_ranges.append((start, end))
            closed, opened = _split_value(text, start, end, spans, entity_type)
            results.append({
                "kind": kind,
                "type": entity_type,
                "value": value,
                "coverage": _coverage(start, end, spans, entity_type),
                "closed": closed,
                "opened": opened,
            })

    extras = [
        span for span in spans
        if not any(span["start"] >= start and span["end"] <= end for start, end in expected_ranges)
    ]

    violations = []
    for value in case["forbidden"]:
        start = text.find(value)
        if start < 0:
            continue
        end = start + len(value)
        covering = [span for span in spans if span["start"] < end and span["end"] > start]
        if covering:
            violations.append((value, covering))

    return {
        "results": results,
        "extras": extras,
        "violations": violations,
        "spans": spans,
        "masked": anonymize(text, spans),
    }


def _verdict(item: dict) -> tuple[str, list[str], bool]:
    """Статус, пояснения под ним и признак провала."""
    value, kind, coverage = item["value"], item["kind"], item["coverage"]

    if coverage is None:
        return ("ошибка кейса", ["такого фрагмента нет в тексте — проверьте сам набор"], True)

    complete = coverage >= PARTIAL_THRESHOLD
    closed = _quote(item["closed"])
    opened = _quote(item["opened"])

    if kind == "optional":
        status = "необязательно: найдено" if complete else (
            "необязательно: найдено частично" if coverage > 0 else "необязательно: не найдено")
        if complete:
            return status, [f"закрыто: {closed} → <{item['type']}>"], False
        if coverage > 0:
            return status, [f"закрыто: {closed} → <{item['type']}>",
                            f"открыто: {opened}"], False
        return status, ["поиск таких данных не входит в обязательные — провалом не считается"], False

    if complete:
        return "найдено", [], False
    if coverage > 0:
        return (f"частично {coverage:.0%}", [
            f"закрыто: {closed} → <{item['type']}>",
            f"открыто: {opened} — осталось в тексте без маски",
        ], True)
    return "пропуск", [f"не замаскировано: {_quote([value])} — осталось в тексте"], True


def main() -> int:
    parser = argparse.ArgumentParser(description="Прогон тест-набора анонимайзера")
    parser.add_argument("--only", default=None, help=f"номер набора (1..{len(CASES)})")
    parser.add_argument("--model-dir", default=None,
                        help="чекпоинт для проверки (по умолчанию models/pii_anonymizer)")
    args = parser.parse_args()

    cases = CASES
    if args.only:
        number = int(args.only)
        if not 1 <= number <= len(CASES):
            parser.error(f"наборов всего {len(CASES)}, а запрошен {number}")
        cases = [CASES[number - 1]]

    print(LEGEND)

    model_label = args.model_dir or DEFAULT_MODEL
    print(f"чекпоинт: {model_label}")

    detector = Detector(model_dir=args.model_dir) if args.model_dir else Detector()
    failures: list[tuple[str, dict, str]] = []
    expectations = 0

    for case in cases:
        report = check_case(detector, case)
        print(f"\n{'=' * WRAP}\n{case['id']}\n{'=' * WRAP}")

        _block("БЫЛО (исходный текст)", case["text"])
        _block("СТАЛО (после маскирования)", report["masked"])

        verdicts = [_verdict(item) for item in report["results"]]
        width = max([len(status) for status, _, _ in verdicts] + [7])
        value_width = 64

        print("\nСВЕРКА ОЖИДАНИЙ:")
        for item, (status, notes, failed) in zip(report["results"], verdicts):
            expectations += 1
            value = item["value"]
            if len(value) > value_width:
                value = value[: value_width - 1] + "…"
            print(f"  {status:<{width}}  {item['type']:<8} «{value}»")
            for note in notes:
                print(f"  {'':<{width}}  {note}")
            if failed:
                failures.append((case["id"], item, status))

        if report["extras"]:
            print("\n  закрыто сверх ожиданий (это не ошибка, просто для сведения):")
            for span in report["extras"]:
                print(f"    {span['type']:<8} «{span['text']}»")

        for value, covering in report["violations"]:
            spans_text = ", ".join(f"{span['type']} «{span['text']}»" for span in covering)
            print(f"\n  ЛОЖНОЕ СРАБАТЫВАНИЕ: «{value}» маскировать нельзя, а оно закрыто как {spans_text}")
            failures.append((case["id"], {"type": covering[0]["type"], "value": value}, "ложное срабатывание"))

    print(f"\n{'=' * WRAP}")
    if failures:
        print(f"ИТОГ: провалов {len(failures)} из {expectations} ожиданий")
        for case_id, item, status in failures:
            print(f"  {case_id}: {item['type']} «{item['value']}» — {status}")
    else:
        print(f"ИТОГ: провалов нет, все {expectations} ожиданий закрыты")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
