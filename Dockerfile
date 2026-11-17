# 1. Берём базовый образ с Python 3.12 (лёгкая версия Alpine)
FROM python:3.12-slim

# 2. Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# 3. Копируем requirements.txt и устанавливаем зависимости
# (делаем это ДО копирования кода, чтобы кэшировать слои Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копируем весь код проекта
COPY src/ ./src/
COPY models/ ./models/

# 5. Открываем порт, на котором работает FastAPI
EXPOSE 8000

# 6. Команда запуска приложения
CMD ["uvicorn", "src.model_api:app", "--host", "0.0.0.0", "--port", "8000"]