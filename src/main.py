# camera demo code

import cv2  #opencv library

cap = cv2.VideoCapture(0) #making camera object
if not cap.isOpened():  #did it connect properly?
    raise RuntimeError("Could not open camera")

print("Press q to quit.")
while True:     #create loop
    ret, frame = cap.read()
    if not ret:    #boolean
        break

    cv2.imshow("camera", frame)   #image in window

cap.release()
cv2.destroyAllWindows()