import serial
import time

arduino = serial.Serial("/dev/ttyACM0",9600)

time.sleep(2)

arduino.write(b"OLA\n")

time.sleep(0.5)

while arduino.in_waiting:
    resposta = arduino.readline().decode(errors="ignore").strip()
    print("Arduino respondeu:", resposta)

arduino.close()
