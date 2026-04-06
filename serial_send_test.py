import serial #just for testing the serial communication with the Arduino, not used in the main code
import time

#change this ofc Arduino -> Tools -> Port -> Arduino Mega -> COM3 or whatever port your Arduinn is on
#and make sure baud rate matches the one in your Arduino code

PORT = "COM3"   
BAUD = 9600

ser = serial.Serial("COM3", 9600, timeout=1) #replace with port and baud rate
time.sleep(2)

commands = ["FORWARD", "LEFT", "RIGHT", "STOP"]

for cmd in commands:
    print(f"Sending: {cmd}")
    ser.write((cmd + "\n").encode("utf-8"))
    time.sleep(1)
#want it to stop after each command so we can see the response from the Arduino in the serial monitor, 
#if you want to send commands faster you can remove the sleep and just read the serial output at the end of the loop
    while ser.in_waiting:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print("Arduino:", line)

ser.close()