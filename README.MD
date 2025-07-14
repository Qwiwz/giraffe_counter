# Система автоматизированного учета жирафов в зоопарке

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-green)](https://flask.palletsprojects.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0.0-red)](https://ultralytics.com/yolov8)

Система для автоматического подсчета жирафов на изображениях с использованием YOLOv8.

## Особенности
- 🦒 Детекция жирафов в реальном времени
- 📊 История запросов с фильтрацией
- 🖼️ Визуализация результатов
- 📈 Статистика обнаружений

## Технологии
- Python 3.13
- Flask
- YOLOv8
- OpenCV
- SQLite

## Установка
```bash
# Клонировать репозиторий
git clone https://github.com/yourusername/giraffe-counter.git
cd giraffe-counter

# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# Установить зависимости
pip install -r requirements.txt

# Запустить приложение
flask run