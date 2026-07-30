from ultralytics import YOLO

model = YOLO('yolov8n.pt')

print("Iniciando detecção...")
results = model.predict(source='https://ultralytics.com/images/bus.jpg', save=True, conf=0.5)

print("Pronto! A imagem foi salva na pasta 'run/detect/predict'")
