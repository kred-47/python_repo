"""Собирает train/val JSONL из parquet russian-pii-66k.

На выходе — записи вида {id, text, entities: [{type, start, end, label}]},
где type — coarse-класс анонимайзера, label — исходная метка датасета.
Ни одна метка не выбрасывается молча: неизвестная метка роняет сборку.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from anonymizer.schema import to_coarse

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "russian_pii_66k" / "train.parquet"
OUT_DIR = ROOT / "data" / "processed" / "russian_pii_66k"
SEED = 42
VAL_SHARE = 0.1


def convert_record(record_id: int, text: str, mask) -> tuple[dict | None, str | None]:
    entities = []
    spans = [dict(span) for span in mask] if mask is not None else []
    ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))

    previous_end = -1
    for span in ordered:
        start, end = int(span["start"]), int(span["end"])
        if start < previous_end:
            return None, f"пересечение спанов на записи {record_id}"
        if text[start:end] != span["value"]:
            return None, f"смещение не совпало с value на записи {record_id}"
        previous_end = end

        try:
            coarse = to_coarse(span["label"])
        except KeyError as exc:
            return None, str(exc)

        entities.append(
            {"type": coarse, "start": start, "end": end, "label": span["label"]}
        )

    return {
        "id": f"russian-pii-66k/{record_id}",
        "text": text,
        "entities": entities,
        "meta": {"language": "ru", "locale": "RU"},
    }, None


def main() -> None:
    df = pd.read_parquet(SRC)
    print(f"прочитано записей: {len(df):,}")

    records: list[dict] = []
    quarantine: list[str] = []
    for record_id, row in enumerate(df.itertuples(index=False)):
        record, error = convert_record(record_id, row.source_text, row.privacy_mask)
        if error:
            quarantine.append(error)
        else:
            records.append(record)

    if quarantine:
        print(f"в карантин: {len(quarantine):,}")
        for message in Counter(quarantine).most_common(5):
            print(f"  {message[1]}x {message[0]}")

    indices = list(range(len(records)))
    random.Random(SEED).shuffle(indices)
    val_size = max(1, int(round(len(records) * VAL_SHARE)))
    val_idx = set(indices[:val_size])

    train = [record for i, record in enumerate(records) if i not in val_idx]
    val = [record for i, record in enumerate(records) if i in val_idx]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, chunk in (("train", train), ("val", val)):
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as out:
            for record in chunk:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{name}: {len(chunk):,} записей -> {path}")

    span_counts: Counter[str] = Counter()
    record_counts: Counter[str] = Counter()
    for record in records:
        for entity in record["entities"]:
            span_counts[entity["type"]] += 1
        for entity_type in {e["type"] for e in record["entities"]}:
            record_counts[entity_type] += 1

    print()
    print(f"{'класс':<10}{'спанов':>9}{'доля записей':>15}")
    print("-" * 34)
    for entity_type, count in span_counts.most_common():
        print(f"{entity_type:<10}{count:>9,}{record_counts[entity_type] / len(records):>14.1%}")

    empty = sum(1 for record in records if not record["entities"])
    print()
    print(f"записей без сущностей: {empty:,}")


if __name__ == "__main__":
    main()
