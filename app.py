import os
import cv2
import numpy as np
import sqlite3
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from datetime import datetime

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
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Детекция только жирафов (класс 23 в COCO)
        results = model.predict(img, classes=[23], conf=0.5)
        
        # Генерируем уникальное имя файла
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        result_filename = f'result_{timestamp}.jpg'
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        
        # Обрабатываем результаты
        if len(results) > 0 and hasattr(results[0], 'plot'):
            result_img = results[0].plot()
            cv2.imwrite(result_path, result_img)
            giraffe_count = len(results[0].boxes)
        else:
            giraffe_count = 0
            cv2.imwrite(result_path, img)  # Сохраняем оригинал, если ничего не найдено
        
        # Сохраняем запрос в базе данных
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        c.execute('''INSERT INTO requests (timestamp, filename, result_img, giraffe_count)
                     VALUES (?, ?, ?, ?)''', 
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                   filename, 
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
                'result_img': os.path.join(app.config['UPLOAD_FOLDER'], row[3]),
                'count': row[4]
            })
        
        return render_template('history.html', history=formatted_history)
    
    except Exception as e:
        return f"Database error: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)