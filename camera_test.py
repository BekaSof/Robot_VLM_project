import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Could not read frame from camera")

cv2.imwrite("frame.jpg", frame)
print("Frame captured and saved as frame.jpg")