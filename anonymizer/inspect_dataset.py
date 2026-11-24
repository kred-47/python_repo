"""Инвентарь датасета russian-pii-66k: метки, целостность смещений, статистика.

Печатает отчёт и ничего не меняет. Проверяет главное: text[start:end] == value
для каждого спана — если это не так, обучаться на смещениях нельзя.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

PARQUET = (
    Path(__file__).resolve().parent.parent
    / "data" / "raw" / "russian_pii_66k" / "train.parquet"
)


def _spans(value) -> list[dict]:
    """privacy_mask приходит как numpy-массив словарей; приводим к списку."""
    if value is None:
        return []
    return [dict(span) for span in value]


def main() -> None:
    df = pd.read_parquet(PARQUET)
    print(f"записей: {len(df):,}")
    print(f"колонки: {list(df.columns)}")
    print(f"языки: {df['language'].value_counts().to_dict()}")
    print(f"локали: {df['locale'].value_counts().to_dict()}")
    print()

    label_counts: Counter[str] = Counter()
    label_records: Counter[str] = Counter()
    mismatches: list[tuple[str, dict, str]] = []
    overlapping = 0
    empty_records = 0
    span_total = 0
    per_record_spans: Counter[int] = Counter()

    for text, mask in zip(df["source_text"], df["privacy_mask"]):
        spans = _spans(mask)
        span_total += len(spans)
        per_record_spans[len(spans)] += 1
        if not spans:
            empty_records += 1
            continue

        seen_labels: set[str] = set()
        ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))

        for span in ordered:
            label = span["label"]
            label_counts[label] += 1
            seen_labels.add(label)

            actual = text[span["start"]:span["end"]]
            if actual != span["value"] and len(mismatches) < 15:
                mismatches.append((label, span, actual))

        for prev, current in zip(ordered, ordered[1:]):
            if current["start"] < prev["end"]:
                overlapping += 1

        for label in seen_labels:
            label_records[label] += 1

    total = len(df)
    print(f"спанов всего: {span_total:,}")
    print(f"записей без сущностей: {empty_records:,} ({empty_records / total:.1%})")
    print(f"спанов на запись: {dict(sorted(per_record_spans.items())[:8])}")
    print()
    print(f"{'метка':<22}{'спанов':>9}{'доля записей':>15}")
    print("-" * 46)
    for label, count in label_counts.most_common():
        share = label_records[label] / total
        print(f"{label:<22}{count:>9,}{share:>14.1%}")
    print()

    unique_labels = sorted(label_counts)
    print(f"уникальных меток: {len(unique_labels)}")
    print(", ".join(unique_labels))
    print()

    bad = sum(1 for _ in mismatches)
    print(f"нарушений смещений (text[start:end] != value): "
          f"{bad}{'+' if bad >= 15 else ''}")
    for label, span, actual in mismatches[:10]:
        print(f"  {label}: [{span['start']}:{span['end']}] "
              f"value={span['value']!r} факт={actual!r}")
    print(f"пересечений между спанами: {overlapping}")


if __name__ == "__main__":
    main()
