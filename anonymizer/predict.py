"""Инференс анонимайзера: загрузка модели, поиск ПДн, замена.

CLI:
    python -m anonymizer.predict "Текст с ПДн"
    python -m anonymizer.predict --mode tokenize "Текст с ПДн"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from anonymizer.decode import spans_from_labels
from anonymizer.merge import merge_entities
from anonymizer.regex_layer import find_regex_entities

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "pii_anonymizer"


class Detector:
    """Обёртка над чекпоинтом: грузим один раз, предсказываем многократно."""

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL, device: str | None = None,
                 max_length: int = 128, overlap: int = 32) -> None:
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_length = max_length
        self.overlap = overlap
        if not 0 <= overlap < max_length - 2:
            raise ValueError("overlap должен быть меньше окна и не отрицательным")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()
        # метки берём из чекпоинта, а не из констант — иначе порядок разъедется
        self.id2label: dict[int, str] = dict(self.model.config.id2label)

    def _ner_spans(self, text: str) -> list[dict]:
        """Скользящее окно по токенам: длинный текст не обрезается, а режется."""
        encoded = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        token_ids: list[int] = encoded["input_ids"]
        offsets: list[tuple[int, int]] = encoded["offset_mapping"]

        window = max(8, self.max_length - 2)  # место под [CLS] и [SEP]
        step = max(1, window - self.overlap)
        found: set[tuple[str, int, int]] = set()

        for start in range(0, max(1, len(token_ids)), step):
            chunk_ids = token_ids[start:start + window]
            if not chunk_ids:
                break
            chunk_offsets = offsets[start:start + window]

            ids = [self.tokenizer.cls_token_id, *chunk_ids, self.tokenizer.sep_token_id]
            input_ids = torch.tensor([ids], device=self.device)
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits

            # отрезаем [CLS] слева, [SEP] не попадает из-за длины chunk_ids
            predicted = logits.argmax(dim=-1)[0][1:1 + len(chunk_ids)].tolist()
            found |= spans_from_labels(predicted, chunk_offsets, text, self.id2label)

            if start + window >= len(token_ids):
                break

        return [
            {"type": entity_type, "start": start, "end": end, "text": text[start:end]}
            for entity_type, start, end in found
        ]

    def detect(self, text: str) -> list[dict]:
        ner = self._ner_spans(text)
        # структурные типы надёжнее ловятся формулой — regex имеет приоритет
        return merge_entities(text, ner, find_regex_entities(text))


def anonymize(text: str, entities: list[dict], mode: str = "mask") -> str:
    """Заменяет спаны справа налево, чтобы не сбить смещения."""
    ordered = sorted(entities, key=lambda item: item["start"], reverse=True)

    if mode == "mask":
        result = text
        for entity in ordered:
            result = result[: entity["start"]] + f"<{entity['type']}>" + result[entity["end"]:]
        return result

    if mode == "tokenize":
        result = text
        for index, entity in enumerate(ordered):
            result = (
                result[: entity["start"]]
                + f"⟦{entity['type']}_{index}⟧"
                + result[entity["end"]:]
            )
        return result

    raise ValueError(f"неизвестный режим замены: {mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Поиск и замена ПДн в тексте")
    parser.add_argument("text", nargs="+", help="текст для обработки")
    parser.add_argument("--mode", default="mask", choices=["mask", "tokenize"])
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", default=None, choices=["auto", "mps", "cpu"])
    args = parser.parse_args()

    device = None if args.device in (None, "auto") else args.device
    detector = Detector(args.model_dir, device=device)
    text = " ".join(args.text)

    entities = detector.detect(text)
    print(f"найдено сущностей: {len(entities)}")
    for entity in entities:
        print(f"  {entity['type']:<10} {entity['text']!r}")
    print()
    print(anonymize(text, entities, args.mode))


if __name__ == "__main__":
    main()
