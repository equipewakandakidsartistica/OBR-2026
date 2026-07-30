import cv2
from ultralytics import YOLO

model = YOLO("/home/wakandakidsart/Documentos/2CAM/runs/detect/train-13/weights/best.pt")

cap =cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("Iniciando identificação")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a câmera.")
        break

    results = model.predict(source=frame, show=True, conf=0.6, verbose=False)

    for r in results:
        if len(r.boxes) > 0:
            for box in r.boxes:
                nome = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                print(f"ALERTA: Identifiquei um(a) {nome} com {conf*100:.1f}% de certeza!")

        frame_desenhado = r.plot()

    cv2.imshow("Identificação", frame_desenhado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows
