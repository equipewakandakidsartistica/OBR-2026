import cv2
from ultralytics import YOLO
import time

# Modelos
model_qr = YOLO("/home/wakandakidsart/Documentos/2CAM/runs/detect/train-13/weights/best.pt")
model_mov = YOLO("/home/wakandakidsart/Documentos/2CAM/runs/detect/train-12/weights/best.pt")

# Duas câmeras
cap_qr = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap_mov = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)

for cap in [cap_qr, cap_mov]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Sistema com 2 câmeras iniciando...")

ultimo_qr = False

while True:
    ret_qr, frame_qr = cap_qr.read()
    ret_mov, frame_mov = cap_mov.read()

    if not ret_qr:
        print("Erro câmera QR")
        break

    if not ret_mov:
        print("Erro câmera MOV")
        break

    results_qr = model_qr(frame_qr, conf=0.6, verbose=False)

    qr_detectado = False

    for r in results_qr:
        if len(r.boxes) > 0:
            qr_detectado = True

            for box in r.boxes:
                nome = model_qr.names[int(box.cls[0])]

                if not ultimo_qr:
                    print(f"[QR DETECTADO] {nome}")

    if not qr_detectado:
        results_mov = model_mov(frame_mov, conf=0.6, verbose=False)

        for r in results_mov:
            if len(r.boxes) > 0:
                for box in r.boxes:
                    nome = model_mov.names[int(box.cls[0])]
                    print(f"[MOV DETECTADO] {nome}")

    ultimo_qr = qr_detectado

cap_qr.release()
cap_mov.release()
cv2.destroyAllWindows()
