from datasets import load_dataset
import pandas as pd
from pathlib import Path

def download_and_save_dataset():
    # Загружаем датасет отзывов с Кинопоиска
    print("Загружаем датасет с Кинопоиска...")
    dataset = load_dataset("blinoff/kinopoisk")

    # Берем только обучающую часть (train)
    train_data = dataset["train"]

    # Превращаем в pandas DataFrame для удобства
    df = pd.DataFrame(train_data)

    # Создаём словарь соответствия
    grades_to_labels = {
        'Good': 'positive',
        'Neutral': 'neutral',
        'Bad': 'negative'
    }

    print("Уникальные значения grade3:", df["grade3"].unique())

    df["label"] = df["grade3"].map(grades_to_labels)

    df = df[["content", "label"]].rename(columns={"content": "text"})

    # Смотрим, что у нас получилось
    print(f"📊 Размер датасета: {len(df)} отзывов")
    print(f"📋 Колонки: {df.columns.tolist()}")
    print(f"🎯 Распределение меток:\n{df['label'].value_counts()}")

    # Сохраняем в CSV
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "kinopoisk_reviews.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"💾 Датасет сохранён в {csv_path}")

    return df

if __name__ == "__main__":
    download_and_save_dataset()