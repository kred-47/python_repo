import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def train_and_save_model():
    # 1. Учебные данные (текст и метка)
    texts = [
        "Отличный продукт, очень доволен!",
        "Ужасное качество, деньги на ветер.",
        "Нормально, но доставка долгая.",
        "Превосходно! Рекомендую всем.",
        "Полный развод, не покупайте.",
        "Хорошая погода сегодня"
    ]
    labels = ["positive", "negative", "neutral", "positive", "negative", "positive"]

    # 2. Создаем конвейер: векторизация текста + классификатор
    model = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", LogisticRegression())
    ])

    # 3. Обучаем модель
    model.fit(texts, labels)
    print("✅ Модель обучена!")

    # 4. Сохраняем модель в файл
    joblib.dump(model, "models/sentiment_model.pkl")
    print("💾 Модель сохранена в models/sentiment_model.pkl")

if __name__ == "__main__":
    # Создаем папку models, если её нет
    os.makedirs("models", exist_ok=True)

    train_and_save_model()