import cv2
import time
import threading
import serial
from ultralytics import YOLO

# ==========================
# MODELOS YOLO
# ==========================
model_qr = YOLO("./runs/detect/train-13/weights/best.pt")
model_mov = YOLO("./runs/detect/train-12/weights/best.pt")

# ==========================
# COMUNICAÇÃO COM ARDUINO
# ==========================
try:
    arduino = serial.Serial("/dev/ttyACM0", 9600, timeout=1)

    time.sleep(3)

    arduino.reset_input_buffer()
    arduino.reset_output_buffer()

    print("Arduino conectado.")

except Exception as e:

    arduino = None

    print("Arduino não encontrado.")
    print(e)

# ==========================
# CÂMERAS
# ==========================
cap_qr = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap_mov = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)

for cap in [cap_qr, cap_mov]:

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*'MJPG')
    )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

# ==========================
# VARIÁVEIS GLOBAIS
# ==========================
frame_qr = None
frame_mov = None

lock_qr = threading.Lock()
lock_mov = threading.Lock()

# ==========================
# THREAD CAMERA QR
# ==========================
def captura_qr():

    global frame_qr

    while True:

        ret, frame = cap_qr.read()

        if ret:
            with lock_qr:
                frame_qr = frame

        time.sleep(0.001)

# ==========================
# THREAD CAMERA MOVIMENTO
# ==========================
def captura_mov():

    global frame_mov

    while True:

        ret, frame = cap_mov.read()

        if ret:
            with lock_mov:
                frame_mov = frame

        time.sleep(0.001)

# ==========================
# INICIAR THREADS
# ==========================
threading.Thread(
    target=captura_qr,
    daemon=True
).start()

threading.Thread(
    target=captura_mov,
    daemon=True
).start()

print("=================================")
print(" Sistema iniciado ")
print("=================================")

frame_count = 0
qr_ativo = False
ultimo_comando = ""

# ======================================
# LOOP PRINCIPAL
# ======================================

while True:

    # Aguarda as duas câmeras iniciarem
    with lock_qr:
        img_qr = None if frame_qr is None else frame_qr.copy()

    with lock_mov:
        img_mov = None if frame_mov is None else frame_mov.copy()

    if img_qr is None or img_mov is None:
        time.sleep(0.1)
        continue

    frame_count += 1

    qr_detectado = False

    # ======================================
    # DETECÇÃO DE QR
    # ======================================

    if frame_count % 2 == 0:

        results_qr = model_qr.predict(
            img_qr,
            conf=0.70,
            verbose=False
        )

        for r in results_qr:

            if len(r.boxes) == 0:
                continue

            qr_detectado = True

            if not qr_ativo:

                print(">>> QR DETECTADO")

                if arduino is not None and ultimo_comando != "QR":

                    arduino.write(b"QR\n")
                    arduino.flush()

                    print("Enviado -> QR")

                    ultimo_comando = "QR"

            for box in r.boxes:

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                classe = int(box.cls[0])

                nome = model_qr.names[classe]

                conf = float(box.conf[0])

                cv2.rectangle(
                    img_qr,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    img_qr,
                    f"{nome} {conf:.2f}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

    # ======================================
    # DETECÇÃO DE MOVIMENTO
    # ======================================

    if not qr_detectado:

        if qr_ativo:

            print(">>> QR SUMIU")

            if arduino is not None:

                arduino.write(b"STOP\n")
                arduino.flush()

                print("Enviado -> STOP")

                ultimo_comando = "STOP"

        if frame_count % 4 == 0:

            results_mov = model_mov.predict(
                img_mov,
                conf=0.70,
                verbose=False
            )

            for r in results_mov:

                if len(r.boxes) == 0:
                    continue

                for box in r.boxes:

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    classe = int(box.cls[0])

                    nome = model_mov.names[classe]

                    conf = float(box.conf[0])

                    print(f"[MOV] {nome}")

                    cv2.rectangle(
                        img_mov,
                        (x1,y1),
                        (x2,y2),
                        (255,0,0),
                        2
                    )

                    cv2.putText(
                        img_mov,
                        f"{nome} {conf:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,0,0),
                        2
                    )

                    if arduino is not None:

                        comando = f"MOV:{nome}"

                        if comando != ultimo_comando:

                            arduino.write((comando+"\n").encode())
                            arduino.flush()

                            print(f"Enviado -> {comando}")

                            ultimo_comando = comando

    qr_ativo = qr_detectado

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======================================
# ENCERRAMENTO
# ======================================

print("\nEncerrando sistema...")

try:

    if arduino is not None:

        arduino.write(b"STOP\n")
        arduino.flush()

        time.sleep(0.2)

        arduino.close()

except:
    pass

try:
    cap_qr.release()
except:
    pass

try:
    cap_mov.release()
except:
    pass

cv2.destroyAllWindows()

print("Sistema encerrado.")
