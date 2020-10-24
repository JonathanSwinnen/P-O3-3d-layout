import cv2
import numpy as np
from vedo import Plotter

cam = cv2.VideoCapture(1)
cam_2 = cv2.VideoCapture(2)

cv2.namedWindow("Frame")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
#    cv2.imshow("test", frame)
    
    ret_2, frame_2 = cam_2.read()
    if not ret_2:
        print("failed to grab frame_2")
        break
#    cv2.imshow("test_2", frame_2)
    both = np.concatenate((frame, frame_2), axis=1)
    cv2.imshow('Frame', both)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        print(frame)
        img_name = "R{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_name_2 = "L{}.png".format(img_counter)
        cv2.imwrite(img_name_2, frame_2)
        print("{} written!".format(img_name_2))
        img_counter += 1

cam.release()
cam_2.release()

cv2.destroyAllWindows()