"""Замер анонимайзера на размеченном наборе: recall, precision и утечки по классам.

Набор — JSONL, по записи на строку:
    {"id": "...", "text": "...", "spans": [{"type": "PER", "start": 3, "end": 27}]}

Запуск:
    python -m anonymizer.eval --data data/processed/nerus/fragments.jsonl
    python -m anonymizer.eval --data ... --model-dir models/pii_anonymizer --out metrics.json

Считается три уровня:
    exact   — предсказанный спан совпал с gold по границам;
    covered — gold закрыт целиком, границы могут быть шире;
    missed  — не закрыто ничего (это и есть утечка).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from anonymizer.predict import DEFAULT_MODEL, Detector
from anonymizer.schema import ENTITY_TYPES


def _intervals(spans: list[dict], entity_type: str | None = None) -> list[tuple[int, int]]:
    return [
        (span["start"], span["end"])
        for span in spans
        if entity_type is None or span["type"] == entity_type
    ]


def _covered_chars(start: int, end: int, spans: list[dict], entity_type: str | None) -> int:
    """Сколько символов диапазона закрыто спанами нужного типа (без двойного счёта)."""
    clipped = sorted(
        (max(start, s), min(end, e))
        for s, e in _intervals(spans, entity_type)
        if s < end and e > start
    )
    total, cursor = 0, start
    for s, e in clipped:
        if e <= cursor:
            continue
        total += e - max(s, cursor)
        cursor = max(cursor, e)
    return total


def _overlaps(start: int, end: int, spans: list[dict], entity_type: str | None) -> bool:
    return any(s < end and e > start for s, e in _intervals(spans, entity_type))


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record.setdefault("id", f"row-{len(records)}")
        records.append(record)
    return records


def evaluate(detector: Detector, records: list[dict]) -> dict:
    stats: dict[str, dict[str, int]] = {}
    conflict: dict[str, int] = {}
    misses: list[dict] = []

    for record in records:
        text = record["text"]
        gold = record["spans"]
        predicted = detector.detect(text)

        gold_types = {span["type"] for span in gold}
        predicted_types = {span["type"] for span in predicted}

        for entity_type in gold_types | predicted_types:
            bucket = stats.setdefault(entity_type, {
                "gold": 0, "exact": 0, "covered": 0, "partial": 0, "missed": 0,
                "chars": 0, "chars_covered": 0,
                "predicted": 0, "predicted_exact": 0, "predicted_loose": 0,
            })
            conflict.setdefault(entity_type, 0)

            for span in (g for g in gold if g["type"] == entity_type):
                start, end = span["start"], span["end"]
                length = end - start
                hit = _covered_chars(start, end, predicted, entity_type)
                bucket["gold"] += 1
                bucket["chars"] += length
                bucket["chars_covered"] += hit

                exact = any(s == start and e == end for s, e in _intervals(predicted, entity_type))
                if exact:
                    bucket["exact"] += 1
                    bucket["covered"] += 1
                elif hit == length:
                    bucket["covered"] += 1
                elif hit:
                    bucket["partial"] += 1
                    misses.append({
                        "id": record["id"], "type": entity_type, "kind": "частично",
                        "gold": text[start:end],
                    })
                else:
                    bucket["missed"] += 1
                    near = [span["text"] for span in predicted
                            if _overlaps(span["start"], span["end"], [{"start": start, "end": end}], None)]
                    misses.append({
                        "id": record["id"], "type": entity_type, "kind": "не найдено",
                        "gold": text[start:end], "predicted": near[:3],
                    })

            for span in (p for p in predicted if p["type"] == entity_type):
                bucket["predicted"] += 1
                if any(span["start"] == g["start"] and span["end"] == g["end"]
                       for g in gold if g["type"] == entity_type):
                    bucket["predicted_exact"] += 1
                elif _overlaps(span["start"], span["end"], gold, entity_type):
                    bucket["predicted_loose"] += 1
                elif _overlaps(span["start"], span["end"], gold, None):
                    conflict[entity_type] += 1

    micro = {"gold": 0, "exact": 0, "missed": 0, "chars": 0, "chars_covered": 0,
             "predicted": 0, "predicted_exact": 0, "predicted_loose": 0}
    for bucket in stats.values():
        for key in micro:
            micro[key] += bucket[key]

    return {"classes": stats, "conflict": conflict, "micro": micro,
            "misses": misses, "fragments": len(records)}


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Замер анонимайзера на размеченном наборе")
    parser.add_argument("--data", required=True, help="JSONL с text и spans")
    parser.add_argument("--model-dir", default=None, help="чекпоинт (по умолчанию models/pii_anonymizer)")
    parser.add_argument("--out", default=None, help="куда сохранить метрики в JSON")
    parser.add_argument("--limit", type=int, default=0, help="ограничить число фрагментов")
    parser.add_argument("--misses", type=int, default=10, help="сколько промахов напечатать")
    args = parser.parse_args()

    path = Path(args.data)
    records = load_records(path)
    if args.limit:
        records = records[: args.limit]

    model_label = args.model_dir or DEFAULT_MODEL
    print(f"чекпоинт: {model_label}")
    print(f"фрагментов: {len(records)}")

    report = evaluate(Detector(model_dir=args.model_dir), records)
    classes = report["classes"]
    order = [t for t in ENTITY_TYPES if t in classes] + sorted(set(classes) - set(ENTITY_TYPES))

    print(f"\n{'класс':<9}{'gold':>6}{'exact':>7}{'покр':>6}{'част':>6}{'проп':>6}"
          f"{'R_exact':>9}{'R_char':>8}{'предск':>8}{'P':>7}{'конфл':>7}")
    for entity_type in order:
        bucket = classes[entity_type]
        print(f"{entity_type:<9}{bucket['gold']:>6}{bucket['exact']:>7}{bucket['covered']:>6}"
              f"{bucket['partial']:>6}{bucket['missed']:>6}"
              f"{_ratio(bucket['exact'], bucket['gold']):>9.3f}"
              f"{_ratio(bucket['chars_covered'], bucket['chars']):>8.3f}"
              f"{bucket['predicted']:>8}"
              f"{_ratio(bucket['predicted_exact'] + bucket['predicted_loose'], bucket['predicted']):>7.3f}"
              f"{report['conflict'][entity_type]:>7}")

    micro = report["micro"]
    print(f"\nитого: gold {micro['gold']}, точных {micro['exact']}, утечек {micro['missed']}"
          f" ({_ratio(micro['missed'], max(1, micro['gold'])):.1%})")
    print(f"recall exact {_ratio(micro['exact'], micro['gold']):.3f}, "
          f"recall по символам {_ratio(micro['chars_covered'], micro['chars']):.3f}, "
          f"precision {_ratio(micro['predicted_exact'] + micro['predicted_loose'], micro['predicted']):.3f}")

    if report["misses"] and args.misses:
        print("\nпромахи:")
        for item in report["misses"][: args.misses]:
            near = f" (рядом: {', '.join(item['predicted'])})" if item.get("predicted") else ""
            print(f"  {item['type']:<8} {item['kind']:<12} «{item['gold']}»{near}")

    out = Path(args.out) if args.out else path.parent / f"metrics-{Path(str(model_label)).name}.json"
    out.write_text(json.dumps({"model": str(model_label), "data": str(path), **report},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nметрики: {out}")

    return 0 if micro["missed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
