from ultralytics import YOLO

# Модель автоматически скачается при первом запуске
model = YOLO('yolov8n.pt')
model.export(format='torchscript')  # Конвертация для оптимизации