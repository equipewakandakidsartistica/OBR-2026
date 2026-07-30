import cv2
import time
import threading
from ultralytics import YOLO

# MODELOS
model_qr = YOLO("./runs/detect/train-13/weights/best.pt")
model_mov = YOLO("./runs/detect/train-12/weights/best.pt")

# CÂMERAS
cap_qr = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap_mov = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)

# CONFIGURAÇÂO
cap_qr.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap_mov.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

cap_qr.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_qr.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap_mov.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_mov.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# THREADS DE CAPTURA
frame_qr = None
frame_mov = None

def captura_qr():
    global frame_qr
    while True:
        ret, frame = cap_qr.read()
        if ret:
            frame_qr = frame

def captura_mov():
    global frame_mov
    while True:
        ret, frame = cap_mov.read()
        if ret:
            frame_mov = frame

threading.Thread(target=captura_qr, daemon=True).start()
threading.Thread(target=captura_mov, daemon=True).start()

print("Sistema iniciado")

# CONTROLE

frame_count = 0
qr_ativo = False

while True:
    if frame_qr is None or frame_mov is None:
        continue

    frame_count += 1

    qr_detectado = False

# QR
    if frame_count % 2 == 0:
        results_qr = model_qr(frame_qr, conf=0.7, verbose=False)

        for r in results_qr:
            if len(r.boxes) > 0:
                qr_detectado = True

                if not qr_ativo:
                    print(">>> QR DETECTADO")

                for box in r.boxes:
                    id_classe = int(box.cls.item())
                    nome = model_qr.names[id_classe]
                    print(f"[QR] {nome}")
# Mov
    if not qr_detectado:
       if qr_ativo:
           print(">>> QR SUMIU - VOLTANDO MOV")

       if frame_count % 4 == 0:
           results_mov = model_mov(frame_mov, conf=0.7, verbose=False)

           for r in results_mov:
               if len(r.boxes) > 0:
                   for box in boxes:
                       id_classe = int(box.cls.item())
                       nome = model_mov.names[id_classe]
                       print(f"[MOV] {nome}")

    qr_ativo = qr_detectado

cap_qr.release()
cap_mov.release()
cv2.destroyAllWindows
