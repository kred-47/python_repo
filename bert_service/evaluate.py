import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pathlib import Path


def evaluate_model():
    print("📂 Загружаем данные и модель...")

    # 1. Загружаем модель
    model_path = Path("models/bert_model")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # 2. Загружаем тестовую часть данных (20%)
    df = pd.read_csv("./data/raw/kinopoisk_reviews.csv")
    df = df[df["label"] != "neutral"]
    label_map = {"negative": 0, "positive": 1}
    df["label_id"] = df["label"].map(label_map)

    _, X_test, _, y_test = train_test_split(df["text"], df["label_id"], test_size=0.2, random_state=42)

    predictions = []
    true_labels = y_test.tolist()

    print("🚀 Делаем предсказания (это займет пару минут)...")
    # Проходимся по тестовым данным батчами, чтобы было быстрее
    batch_size = 16
    for i in range(0, len(X_test), batch_size):
        batch_texts = X_test[i:i + batch_size].tolist()
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            predictions.extend(preds)

    # 3. Выводим отчет
    print("\n" + "=" * 50)
    print("ОТЧЕТ ПО КАЧЕСТВУ МОДЕЛИ (BERT)")
    print("=" * 50)
    print(classification_report(true_labels, predictions, target_names=["negative", "positive"]))


if __name__ == "__main__":
    evaluate_model()