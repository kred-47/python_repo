FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Копируем requirements из папки bert_service
COPY bert_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем только код BERT сервиса и общую папку моделей
COPY bert_service/main.py .

# Копируем модели из корневой папки models
COPY ../models/ ./models/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]