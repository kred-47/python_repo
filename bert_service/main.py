import os
# Обязательно для Mac, чтобы не падал при сложных операциях
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from pathlib import Path

app = FastAPI(title="RuModernBERT Sentiment API")

class TextInput(BaseModel):
    text: str

# Загружаем модель и токенизатор ОДИН раз при старте
model_path = Path("models/bert_model")

# Если не нашли (на всякий случай), ищем относительно расположения файла
if not model_path.exists():
    model_path = Path(__file__).parent.parent / "models" / "bert_model"

print("🤖 Загружаем RuModernBERT...")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    attn_implementation="eager",
    trust_remote_code=True
)

# Определяем устройство (с защитой от ложного срабатывания MPS в Docker)
try:
    if torch.backends.mps.is_available():
        # Тестовая проверка: действительно ли эта сборка PyTorch поддерживает MPS
        torch.tensor([1.0]).to("mps")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
except RuntimeError:
    # Если это CPU-only сборка (как в нашем Dockerfile), принудительно используем CPU
    device = torch.device("cpu")

model.to(device)
model.eval() # Переводим в режим инференса (предсказания)
print(f"RuModernBERT загружен на устройство: {device}")

@app.post("/predict")
def predict(data: TextInput):
    # Токенизируем входной текст
    inputs = (tokenizer(
        data.text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device))

    # Делаем предсказание
    with torch.no_grad():
        outputs = model(**inputs)

    # Превращаем логиты в вероятности (Softmax)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    # Классы: 0 - negative, 1 - positive
    classes = ["negative", "positive"]
    prediction_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction_idx] * 100)

    return {
        "text": data.text,
        "prediction": classes[prediction_idx],
        "confidence": round(confidence, 2),
        "probabilities": {
            "negative": round(float(probabilities[0] * 100), 2),
            "positive": round(float(probabilities[1] * 100), 2)
        }
    }