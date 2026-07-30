from ultralytics import YOLO
import cv2

model = YOLO("Documentos/2CAM/runs/detect/train-12/weight/best.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("---MONITOR DE MOVIMENTAÇÂO ATIVO---")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.6, verbose=False)

    for r in results:
        if len(r.boxes) > 0:
            for box in r.boxes:

                nome = model.names[int(box.cls[0])]
                conf = float(box.conf[0])

                print(f"Detectado: {nome} ({conf*100:.1f}%)")

        frame_anotado = r.plot()

    cv2.imshow("Sistema de Movimentação", frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
