# python_repo

Учебный проект: обучение ML-модели на scikit-learn и отдача её через FastAPI.

Модель — `Pipeline` из `TfidfVectorizer` + `LogisticRegression`, классифицирует текст
по тональности: `positive` / `negative` / `neutral`.

## Структура

```
src/model_train.py   обучение модели и сохранение в models/sentiment_model.pkl
src/model_api.py     FastAPI-сервис, загружает модель и отдаёт /predict
src/data_loader.py   скачивает датасет отзывов в data/raw/kinopoisk_reviews.csv
models/              сохранённая модель (лежит в репозитории, пересобирается скриптом)
requirements.txt     зависимости
```

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Обучить модель (запускать из корня проекта — путь `models/` в скрипте относительный):

```bash
python src/model_train.py
```

Поднять API (тоже из корня):

```bash
uvicorn src.model_api:app --reload
```

## Данные

Датасет отзывов Кинопоиска (~590 тыс. строк, 127 МБ) в репозиторий не входит — он
скачивается скриптом:

```bash
python src/data_loader.py
```

Результат — `data/raw/kinopoisk_reviews.csv` (колонки `text`, `label`), папка в `.gitignore`.

## Эндпоинты

`GET /` — проверка, что сервис жив:

```json
{"message": "ML API is running!"}
```

`POST /predict` — предсказание по тексту:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Отличный продукт, очень доволен!"}'
```

Ответ:

```json
{
  "text": "Отличный продукт, очень доволен!",
  "prediction": "positive",
  "confidence": 78.4,
  "probabilities": {"negative": 5.1, "neutral": 16.5, "positive": 78.4}
}
```

`probabilities` — проценты, округлённые до двух знаков.

## Замечания

- Модель загружается один раз при старте приложения, поэтому `.pkl` должен уже лежать
  в `models/` — иначе импорт `model_api` упадёт.
- `models/sentiment_model.pkl` (4 КБ) лежит в репозитории намеренно: после клона API
  запускается без обучения. Когда модель станет настоящей и потяжелеет, её место — вне
  репозитория.
- Обучающая выборка в `model_train.py` — 6 примеров, это демо пайплайна, а не рабочая модель.
