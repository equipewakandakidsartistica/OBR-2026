import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train-13/weights/best.pt")

cap = cv2.VideoCapture(0)

print("Pressione 'q' no teclado para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a câmera")
        break

        results = model(frame, verbose=False, conf=0.5)

        for r in results:
            boxes = r. boxes
            for box in boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cls = int(box.cls[0])
                nome = model.names[cls]
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{nome} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Detecção Unica - 2CAM", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

cap.release()
cv2.destroyAllWindows
