# Команды проекта. Полный список: make help
PY ?= .venv/bin/python
PORT ?= 8002
MODEL ?= models/pii_anonymizer
TEXT ?= Здравствуйте, меня зовут Александр Петров, телефон +7 (916) 123-45-67

.PHONY: help install data inspect dataset prepare train check predict api docker-up docker-down

help:
	@echo "Установка и данные:"
	@echo "  make install    создать .venv и поставить зависимости (сервисы + анонимайзер)"
	@echo "  make data       скачать датасет russian-pii-66k в data/raw"
	@echo "  make inspect    отчёт по меткам и целостности смещений"
	@echo "  make dataset    собрать train/val JSONL в data/processed"
	@echo "  make prepare    data + inspect + dataset одной командой"
	@echo ""
	@echo "Обучение и проверка:"
	@echo "  make train      дообучить модель (epochs=$(EPOCHS), по умолчанию 1) и сохранить чекпоинт"
	@echo "  make check      прогнать тест-набор из 4 проб (MODEL=$(MODEL)), код 1 при провалах"
	@echo ""
	@echo "Запуск:"
	@echo "  make predict TEXT=\"текст\"   разовый прогон в консоли"
	@echo "  make api                    uvicorn на 127.0.0.1:$(PORT)"
	@echo "  make docker-up / docker-down  поднять или остановить оба сервиса в Docker"

install:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt -r anonymizer/requirements.txt

data:
	$(PY) -m anonymizer.download_data

inspect:
	$(PY) -m anonymizer.inspect_dataset

dataset:
	$(PY) -m anonymizer.build_dataset

prepare: data inspect dataset

EPOCHS ?= 1
train:
	$(PY) -m anonymizer.train --epochs $(EPOCHS) --save

check:
	$(PY) -m anonymizer.check --model-dir $(MODEL)

predict:
	$(PY) -m anonymizer.predict --model-dir $(MODEL) "$(TEXT)"

api:
	$(PY) -m uvicorn anonymizer.api:app --host 127.0.0.1 --port $(PORT)

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down
