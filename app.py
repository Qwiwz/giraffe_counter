import os
import cv2
import numpy as np
import sqlite3
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from datetime import datetime
import uuid  # Для генерации уникальных имен файлов

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/results'

# Создаем необходимые директории
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        filename TEXT,
        original_img TEXT,
        thumbnail TEXT,
        result_img TEXT,
        giraffe_count INTEGER
    )''')
    conn.commit()
    conn.close()

# Загрузка модели YOLO
model = YOLO('yolov8n.pt')

# Инициализируем базу данных при запуске
init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_giraffes():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    try:
        file = request.files['image']
        filename = file.filename
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        original_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Детекция только жирафов (класс 23 в COCO)
        results = model.predict(original_img, classes=[23], conf=0.5)
        
        # Генерируем уникальный идентификатор для файлов
        unique_id = uuid.uuid4().hex
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Сохраняем оригинальное изображение
        original_filename = f'original_{timestamp}_{unique_id}.jpg'
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        cv2.imwrite(original_path, original_img)
        
        # Создаем миниатюру оригинального изображения
        thumbnail_filename = f'thumbnail_{timestamp}_{unique_id}.jpg'
        thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], thumbnail_filename)
        thumbnail_img = cv2.resize(original_img, (300, 200))
        cv2.imwrite(thumbnail_path, thumbnail_img)
        
        # Обрабатываем результаты детекции
        if len(results) > 0 and hasattr(results[0], 'plot'):
            result_img = results[0].plot()
            giraffe_count = len(results[0].boxes)
        else:
            result_img = original_img
            giraffe_count = 0
        
        # Сохраняем изображение с результатами
        result_filename = f'result_{timestamp}_{unique_id}.jpg'
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_img)
        
        # Сохраняем запрос в базе данных
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        c.execute('''INSERT INTO requests (
                        timestamp, 
                        filename, 
                        original_img, 
                        thumbnail, 
                        result_img, 
                        giraffe_count
                     ) VALUES (?, ?, ?, ?, ?, ?)''', 
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                   filename, 
                   original_filename, 
                   thumbnail_filename, 
                   result_filename, 
                   giraffe_count))
        conn.commit()
        conn.close()
        
        return jsonify({
            'count': giraffe_count,
            'result_img': result_path
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def show_history():
    """Функция для отображения истории запросов"""
    try:
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        c.execute("SELECT * FROM requests ORDER BY timestamp DESC")
        history = c.fetchall()
        conn.close()
        
        # Форматируем данные для отображения
        formatted_history = []
        for row in history:
            formatted_history.append({
                'id': row[0],
                'timestamp': row[1],
                'filename': row[2],
                'original_img': row[3],
                'thumbnail': row[4],
                'result_img': row[5],
                'count': row[6]
            })
        
        return render_template('history.html', history=formatted_history)
    
    except Exception as e:
        return f"Database error: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)