import cv2
from ultralytics import YOLO
import time

# modelos
model_qr = YOLO("/home/wakandakidsart/Documentos/2CAM/runs/detect/train-13/weights/best.pt")
model_mov = YOLO("/home//wakandakidsart/Documentos/2CAM/runs/detect/train-12/weights/best.pt")

# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("Sistema iniciando...")

ultimo_qr = 0
TEMPO_PAUSA = 2

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar câmera")
        break

    qr_detectado = False

    # Detecção QR
    results_qr = model_qr.predict(source=frame, conf=0.6, verbose=False)

    for r in results_qr:
        if len(r.boxes) > 0:
           qr_detectado = True
           ultimo_qr = time.time()

           for box in r.boxes:
               nome = model_qr.names[int(box.cls[0])]
               conf = float(box.conf[0])
               print(f"[QR] {nome} ({conf*100:.1f}%)")

    # Detecção Movimento
    if time.time() - ultimo_qr > TEMPO_PAUSA:
        results_mov = model_mov.predict(source=frame, conf=0.6, verbose=False)

    if qr_detectado:
        frame = results_qr[0].plot()
    else:
       if 'results_mov' in locals():
           frame = results_mov[0].plot()

    cv2.imshow("Sistema Unificado", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
