import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

def train_and_save_model():
    print("Загружаем данные из csv...")
    df = pd.read_csv("data/raw/kinopoisk_reviews.csv")
    df = df[df["label"] != "neutral"]

    # X - это наши тексты (признаки), y - это метки (то, что предсказываем)
    X = df["text"]
    y = df["label"]

    # 2. Разделяем данные на обучающую (80%) и тестовую (20%) выборки.
    # Функция train_test_split: Это самая важная строка в ML.
    # Мы отдаем модели 80% данных, чтобы она училась, а 20% прячем.
    # Потом просим модель предсказать эти спрятанные 20%.
    # Если модель просто "запомнит" ответы (переобучится), на спрятанных данных она провалится.
    # Функция random_state=42 нужен, чтобы при каждом запуске разделение было одинаковым
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Обучаем на {len(X_train)} отзывах, тестируем на {len(X_test)}")

    # 3. Создаем pipeline
    model = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", LogisticRegression())
    ])

    # 4. Обучаем модель ТОЛЬКО на обучающей выборке
    model.fit(X_train, y_train)

    # 5. Оцениваем качество на тестовой выборке (которую модель ещё не видела)
    y_pred = model.predict(X_test)

    print("\n--- МЕТРИКИ КАЧЕСТВА ---")
    print(f"Точность: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Отчёт по классам:")
    print(classification_report(y_test, y_pred))

    # 6. Сохраняем обновленную модель
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "sentiment_model.pkl")
    print("💾Новая модель сохранена!")

if __name__ == "__main__":
    train_and_save_model()