import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/results'
model = YOLO('yolov8n.pt')  # Автоматически использует скачанную модель

# Создание папок
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_giraffes():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Детекция только жирафов (класс 23 в COCO)
    results = model.predict(img, classes=[23], conf=0.5)
    
    # Сохранение результата
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{timestamp}.jpg')
    cv2.imwrite(result_path, results[0].plot())  # Автоматическая отрисовка bbox
    
    # Подсчет объектов
    giraffe_count = len(results[0].boxes)
    
    return jsonify({
        'count': giraffe_count,
        'result_img': result_path
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)