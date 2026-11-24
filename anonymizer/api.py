"""HTTP-обёртка анонимайзера: FastAPI, порт по умолчанию 8002.

Запуск:
    .venv/bin/python -m uvicorn anonymizer.api:app --port 8002
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from anonymizer.predict import Detector, anonymize

app = FastAPI(title="PII Anonymizer", version="0.1.0")

# модель грузится один раз при старте, а не на каждый запрос
detector = Detector()


class TextRequest(BaseModel):
    text: str = Field(min_length=1, description="текст для обработки")


class AnonymizeRequest(TextRequest):
    mode: str = Field(default="mask", pattern="^(mask|tokenize)$")


@app.get("/")
def root() -> dict:
    return {
        "message": "PII Anonymizer is running",
        "device": str(detector.device),
        "labels": sorted(set(detector.id2label.values()) - {"O"}),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": str(detector.device)}


@app.post("/detect")
def detect(request: TextRequest) -> dict:
    entities = detector.detect(request.text)
    return {"text": request.text, "entities": entities}


@app.post("/anonymize")
def anonymize_text(request: AnonymizeRequest) -> dict:
    entities = detector.detect(request.text)
    return {
        "original": request.text,
        "anonymized": anonymize(request.text, entities, request.mode),
        "mode": request.mode,
        "entities": entities,
    }
