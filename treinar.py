from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data_qr.yaml",
    epochs=100,
    imgsz=448
)