"""Скачивает исходный датасет wolframko/russian-pii-66k в data/raw.

Идемпотентно: если файл уже есть и размер совпадает, повторно не качает.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

DATASET_URL = (
    "https://huggingface.co/datasets/wolframko/russian-pii-66k"
    "/resolve/main/data/train-00000-of-00001.parquet"
)
EXPECTED_SIZE = 10_684_761
TARGET = Path(__file__).resolve().parent.parent / "data" / "raw" / "russian_pii_66k" / "train.parquet"


def download(url: str = DATASET_URL, target: Path = TARGET) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and target.stat().st_size == EXPECTED_SIZE:
        print(f"уже скачано: {target} ({target.stat().st_size:,} байт)")
        return target

    tmp = target.with_suffix(target.suffix + ".part")
    print(f"качаю {url}")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:6.1f} МБ", end="", flush=True)
    print()

    size = tmp.stat().st_size
    if size != EXPECTED_SIZE:
        raise RuntimeError(f"неожиданный размер: {size:,} вместо {EXPECTED_SIZE:,}")

    os.replace(tmp, target)
    print(f"готово: {target} ({size:,} байт)")
    return target


if __name__ == "__main__":
    download()
