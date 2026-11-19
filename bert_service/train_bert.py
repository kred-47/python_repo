import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset
from pathlib import Path

def train_bert_model():
    print("Загружаем данные...")
    df = pd.read_csv("./data/raw/kinopoisk_reviews.csv")
    df = df[df["label"] != "neutral"]

    # BERT требует, чтобы метками были числа 0 и 1
    label_map = {"negative": 0, "positive": 1}
    df["label_id"] = df["label"].map(label_map)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label_id"], test_size=0.2, random_state=42
    )

    # превращаем pandas в формат HuggingFace
    train_dataset = Dataset.from_pandas(pd.DataFrame({"text": X_train, "label": y_train}))
    test_dataset = Dataset.from_pandas(pd.DataFrame({"text": X_test, "label": y_test}))

    # Загружаем современный RuModernBERTда
    model_name = "deepvk/RuModernBERT-base"
    print(f"🤖 Загружаем модель {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        attn_implementation="eager",
        trust_remote_code=True,
    )


    # Функция токенизации (превращает текст в цифры)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    print("Токенизируем тексты...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Удаляем колонку 'text', она больше не нужна, только 'input_ids' и 'attention_mask'
    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # Настройки обучения
    training_args = TrainingArguments(
        output_dir="./bert_results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        weight_decay=0.01,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Начинаем обучение...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Сохраняем модель в общую папку models/
    save_path = Path("models/bert_model")
    save_path.mkdir(exist_ok=True)
    print(f"Сохраняем модель в {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Готово! Модель обучена и сохранена.")

if __name__ == "__main__":
    train_bert_model()