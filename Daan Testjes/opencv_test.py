import cv2
import numpy as np

cam = cv2.VideoCapture(1)
cam_2 = cv2.VideoCapture(2)

def make_1080p():
    cam.set(3, 1920)
    cam.set(4, 1080)
    cam_2.set(3, 1920)
    cam_2.set(4, 1080)

def make_720p():
    cam.set(3, 1280)
    cam.set(4, 720)
    cam_2.set(3, 1280)
    cam_2.set(4, 720)

def make_480p():
    cam.set(3, 640)
    cam.set(4, 480)
    cam_2.set(3, 640)
    cam_2.set(4, 480)

def change_res(width, height):
    cam.set(3, width)
    cam.set(4, height)
    cam_2.set(3, width)
    cam_2.set(4, height)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

make_1080p()

cv2.namedWindow("Frame")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    frame = rescale_frame(frame, 50)
#    cv2.imshow("test", frame)
    
    ret_2, frame_2 = cam_2.read()
    if not ret_2:
        print("failed to grab frame_2")
        break
    frame_2 = rescale_frame(frame_2, 50)
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
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        img_name_2 = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name_2, frame_2)
        print("{} written!".format(img_name_2))
        img_counter += 1

cam.release()
cam_2.release()

cv2.destroyAllWindows()