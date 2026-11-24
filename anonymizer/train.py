"""Дообучение RuModernBERT-small на размеченном корпусе PII (token classification).

Спаны заданы в символах, поэтому метки ставятся по offset_mapping токенизатора:
первый токен сущности получает B-, остальные — I-. Оценка — по точному
совпадению character-span, как в eval_model.py пилота коллеги.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from anonymizer.schema import ENTITY_TYPES

LABELS: list[str] = ["O"] + [
    f"{prefix}-{entity}" for entity in ENTITY_TYPES for prefix in ("B", "I")
]
LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}

ROOT = Path(__file__).resolve().parent.parent

# Хвостовая пунктуация приклеивается к последнему токену и не относится к сущности.
TRAILING_PUNCT = ".,;:!?"

# Соседние сущности одного типа через пробел или запятую — это один объект.
SEPARATORS = " ,"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def merge_adjacent_entities(text: str, entities: list[dict]) -> list[dict]:
    """Склеивает соседние сущности одного типа, разделённые только пробелами/запятыми.

    В корпусе ФИО размечено пословно (GIVENNAME + SURNAME), адрес — по частям
    (STREET + BUILDINGNUM). Без склейки модель учится закрывать одно слово из
    трёх: «имя = одно слово» — и на живом тексте оставляет часть ФИО открытой.
    """
    ordered = sorted(entities, key=lambda entity: (entity["start"], entity["end"]))
    merged: list[dict] = []
    for entity in ordered:
        if merged and merged[-1]["type"] == entity["type"]:
            gap = text[merged[-1]["end"]:entity["start"]]
            if gap.strip(SEPARATORS) == "":
                merged[-1] = {**merged[-1], "end": entity["end"]}
                continue
        merged.append(dict(entity))
    return merged


def normalize_records(records: list[dict]) -> list[dict]:
    """Приводит записи к обучению: составные сущности становятся одним спаном.

    Делается до создания датасета, потому что gold-спаны для оценки берутся
    из тех же entities — иначе метки и эталон разъедутся.
    """
    return [
        {**record, "entities": merge_adjacent_entities(record["text"], record["entities"])}
        for record in records
    ]


class PiiDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        encoded = self.tokenizer(
            record["text"],
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        # O для обычных токенов, -100 только для спецтокенов: иначе модель
        # не видит негативов и размечает сущности везде
        labels = [
            -100 if token_start == token_end else 0
            for token_start, token_end in offsets
        ]

        for entity in record["entities"]:
            start, end = entity["start"], entity["end"]
            first = True
            for position, (token_start, token_end) in enumerate(offsets):
                if token_start == token_end:  # спецтокены
                    continue
                if token_end <= start or token_start >= end:
                    continue
                prefix = "B-" if first else "I-"
                labels[position] = LABEL2ID[prefix + entity["type"]]
                first = False

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "offsets": offsets,
            "entities": record["entities"],
            "text": record["text"],
        }


def collate(batch: list[dict], pad_token_id: int) -> dict:
    width = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        padding = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        attention_mask.append(item["attention_mask"] + [0] * padding)
        labels.append(item["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
        "offsets": [item["offsets"] for item in batch],
        "entities": [item["entities"] for item in batch],
        "texts": [item["text"] for item in batch],
    }


def spans_from_labels(
    label_ids: list[int], offsets: list[tuple[int, int]], text: str
) -> set[tuple[str, int, int]]:
    """Собирает character-span'ы из последовательности предсказанных меток.

    Токенизатор приклеивает ведущий пробел к токену, поэтому границы спана
    обрезаются по пробелам — иначе каждый спан сдвинут на один символ.
    """
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
        label = "O" if token_start == token_end or label_id < 0 else ID2LABEL[label_id]

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


def evaluate(model, loader, device: torch.device) -> dict:
    model.eval()
    counts: dict[str, dict[str, int]] = {
        entity: {"tp": 0, "fp": 0, "fn": 0} for entity in ENTITY_TYPES
    }
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for batch in loader:
            offsets_batch = batch.pop("offsets")
            gold_batch = batch.pop("entities")
            texts_batch = batch.pop("texts")
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss)
            batches += 1

            predictions = outputs.logits.argmax(dim=-1).cpu().tolist()
            for predicted_ids, offsets, gold_entities, text in zip(
                predictions, offsets_batch, gold_batch, texts_batch
            ):
                predicted = spans_from_labels(predicted_ids, offsets, text)
                gold = {
                    (entity["type"], entity["start"], entity["end"]) for entity in gold_entities
                }
                for entity_type, start, end in gold & predicted:
                    counts[entity_type]["tp"] += 1
                for entity_type, start, end in predicted - gold:
                    counts[entity_type]["fp"] += 1
                for entity_type, start, end in gold - predicted:
                    counts[entity_type]["fn"] += 1

    report = {}
    micro = {"tp": 0, "fp": 0, "fn": 0}
    for entity_type, values in counts.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        for key in micro:
            micro[key] += values[key]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[entity_type] = {
            "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn
        }

    tp, fp, fn = micro["tp"], micro["fp"], micro["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    report["_micro"] = {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "loss": total_loss / max(1, batches),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Дообучение NER-модели на PII-корпусе")
    parser.add_argument("--data", default=str(ROOT / "data/processed/russian_pii_66k/train.jsonl"))
    parser.add_argument("--val-data", default=str(ROOT / "data/processed/russian_pii_66k/val.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "models/pii_anonymizer"))
    parser.add_argument("--model-name", default="deepvk/RuModernBERT-small")
    parser.add_argument("--revision", default="patched-tokenizer")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="ограничить число train-записей")
    parser.add_argument("--val-limit", type=int, default=500)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true", help="сохранить модель в --output-dir")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"устройство: {device}")

    train_records = load_jsonl(Path(args.data))
    if args.limit:
        random.Random(args.seed).shuffle(train_records)
        train_records = train_records[: args.limit]
    val_records = load_jsonl(Path(args.val_data))[: args.val_limit]

    before = sum(len(record["entities"]) for record in train_records)
    train_records = normalize_records(train_records)
    val_records = normalize_records(val_records)
    after = sum(len(record["entities"]) for record in train_records)
    print(f"train: {len(train_records):,} | val: {len(val_records):,}")
    print(f"склейка составных сущностей: спанов {before:,} → {after:,}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        revision=args.revision,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    train_loader = DataLoader(
        PiiDataset(train_records, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        PiiDataset(val_records, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    updates_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    model.train()
    started = time.time()
    step = 0
    running_loss = 0.0

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        for index, batch in enumerate(train_loader, start=1):
            batch.pop("offsets")
            batch.pop("entities")
            batch.pop("texts")
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            running_loss += float(loss)

            if index % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % args.log_every == 0:
                    elapsed = time.time() - started
                    print(f"эпоха {epoch + 1} шаг {step}/{total_steps} "
                          f"loss={running_loss / args.log_every:.4f} {elapsed:.0f}s")
                    running_loss = 0.0

        report = evaluate(model, val_loader, device)
        micro = report["_micro"]
        print(f"\nэпоха {epoch + 1}: loss={micro['loss']:.4f} "
              f"P={micro['precision']:.3f} R={micro['recall']:.3f} F1={micro['f1']:.3f}")
        print(f"{'класс':<10}{'P':>7}{'R':>7}{'F1':>7}{'tp':>7}{'fp':>7}{'fn':>7}")
        for entity_type in ENTITY_TYPES:
            values = report[entity_type]
            print(f"{entity_type:<10}{values['precision']:>7.3f}{values['recall']:>7.3f}"
                  f"{values['f1']:>7.3f}{values['tp']:>7,}{values['fp']:>7,}{values['fn']:>7,}")

        if args.save:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"модель сохранена: {output_dir}")


if __name__ == "__main__":
    main()
