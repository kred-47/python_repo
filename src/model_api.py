from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

# Создайте приложение
app = FastAPI()

# Создайте модель для входных данных (используя Pydantic)
class TextInput(BaseModel):
    text: str

# Загружаем модель ОДИН раз при запуске приложения
model_path=Path(__file__).parent.parent / "models" / "sentiment_model.pkl"
model = joblib.load(model_path)
print("Модель успешно загружена!")

# Создайте endpoint для предсказания
@app.post("/predict")
def predict(data: TextInput):
    # Используем модель для предсказания
    # model.predict() принимает список строк, поэтому передаем [data.text]
    # [0] нужен, чтобы достать первый (и единственный) элемент из списка результатов
    prediction = model.predict([data.text])[0]
    probabilities = model.predict_proba([data.text])[0]

    # Выводим классы и их вероятности
    print(f"Классы: {model.classes_}")
    print(f"Вероятности: {probabilities}")

    # Создаём словарь для наглядности
    prob_dict = dict(zip(model.classes_, probabilities))

    prob_dict_clean = {}

    for key, value in prob_dict.items():
        # 1. Убираем numpy-тип через .item()
        # 2. Умножаем на 100
        # 3. Округляем до 2 знаков
        clean_value = round(value.item() * 100, 2)
        prob_dict_clean[key] = clean_value

    confidence = max(prob_dict_clean.values())

    return {
        "text": data.text,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": prob_dict_clean
    }

# Создайте endpoint для проверки работоспособности
@app.get("/")
def root():
    return {"message": "ML API is running!"}