import cv2
from ultralytics import YOLO
import time

# Modelos
model_qr = YOLO("/home/wakandakidsart/Documentos/2CAM/runs/detect/train-13/weights/best.pt")
model_mov = YOLO("/home/wakandakidsart/Documentos/2CAM/runs/detect/train-12/weights/best.pt")

# CÂMERAS
cap_qr = cv2.VideoCapture(0)
cap_mov = cv2.VideoCapture(1)

# RESOLUÇÃO BAIXA
for cap in [cap_qr, cap_mov]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("Sistema com 2 câmeras iniciando...")

ultime_qr = 0
TEMPO_PAUSA = 2

frame_count = 0

while True:
    ret_qr, frame_qr = cap_qr.read()
    ret_mov, frame_mov = cap_mov.read()

    if not ret_qr or not ret_mov:
        print("Erro nas câmeras")
        break

    qr_detectado = False

    frame_count += 1

#  Detecção
    if frame_count % 2 == 0:
        results_qr = model_qr.predict(frame_qr, conf=0.6, verbose=False)

        for r in results_qr:
            if len(r.boxes) > 0:
                qr_detectado = True
                ultimo_qr = time.time()

                for box in r.boxes:
                    nome = model_qr.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    print(f"[QR {nome} ({conf*100:.1f}%")

    if time.time() - ultimo_qr > TEMPO_PAUSA:
        if frame_count % 3 == 0:
            results_mov = model_mov.predict(frame_mov, conf=0.6, verbose=False)

            for r in results_mov:
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        nome = model_mov.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        print(f"[MOV] {nome} ({conf*100:.1f}%)")

# DESENHO
    if 'results_qr' in locals():
        frame_qr = results_qr[0].plot()

    if 'results_mov' in locals():
        frame_mov = results_mov[0].plot()

# MOSTRAR
    cv2.imshow("Camera QR", frame_qr)
    cv2.imshow("Camera MOV", frame_mov)

cap_qr.release()
cap_mov.release()
cv2.destroyAllWindows()
