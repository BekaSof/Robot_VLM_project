import serial
import time

ser = serial.Serial("COM8", 9600, timeout=1)
time.sleep(2)

for cmd in ["FORWARD", "LEFT", "RIGHT", "BACK", "STOP"]:
    print("Sending:", cmd)
    ser.write((cmd + "\n").encode("utf-8"))
    time.sleep(2)

ser.close()
