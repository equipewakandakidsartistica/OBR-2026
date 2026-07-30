import serial
import time

arduino = serial.Serial("/dev/ttyACM0",9600)

time.sleep(2)

arduino.write(b"QR\n")
print("QR")
time.sleep(5)

arduino.write(b"STOP\n")
print("STOP")
time.sleep(5)

arduino.write(b"MOV:pessoa\n")
print("MOV")
