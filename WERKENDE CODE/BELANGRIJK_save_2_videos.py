import numpy as np
import sys
import cv2

print("connecting cameras")
cap1 = cv2.VideoCapture(1)
print("camera 1 connected")
cap2 = cv2.VideoCapture(2)
print("camera 2 connected")

cap1.set(3, 1920)
cap1.set(4, 1080)
cap2.set(3, 1920)
cap2.set(4, 1080)

# Define the codec and create VideoWriter object
fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('output_L.avi', fourcc1, 30.0, (1920, 1080))
fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
out2 = cv2.VideoWriter('output_R.avi', fourcc2, 30.0, (1920, 1080))

while(cap1.isOpened() and cap2.isOpened()):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret1 and ret2:
        # write the frames
        out1.write(frame1)
        out2.write(frame2)

        cv2.imshow('frame', frame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()