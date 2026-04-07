import serial #just for testing the serial communication with the Arduino, not used in the main code
import time

PORT = "COM8"   # change this
BAUD = 9600

def send_command(ser, cmd: str):
    print(f"Sending: {cmd}")
    ser.write((cmd + "\n").encode("utf-8"))
    time.sleep(0.1)

    end_time = time.time() + 2
    while time.time() < end_time:
        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                print("Arduino:", line)

with serial.Serial(PORT, BAUD, timeout=1) as ser:
    time.sleep(2)  # let Arduino reset

    # clear startup text
    while ser.in_waiting:
        print("Arduino:", ser.readline().decode(errors="ignore").strip())

    send_command(ser, "FORWARD")
    time.sleep(1)

    send_command(ser, "LEFT")
    time.sleep(1)

    send_command(ser, "RIGHT")
    time.sleep(1)

    send_command(ser, "BACK")
    time.sleep(1)

    send_command(ser, "STOP")