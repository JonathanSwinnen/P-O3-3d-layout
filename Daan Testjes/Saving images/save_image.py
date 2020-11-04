import cv2
import numpy as np
import time

print("setting up camera 1")
#cam_1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
print("setting up camera 2")
cam_2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
print("done!")


def make_1080p():
    cam_1.set(3, 1920)
    cam_1.set(4, 1080)
    cam_2.set(3, 1920)
    cam_2.set(4, 1080)


def make_720p():
    cam_1.set(3, 1280)
    cam_1.set(4, 720)
    cam_2.set(3, 1280)
    cam_2.set(4, 720)


def make_480p():
    cam_1.set(3, 640)
    cam_1.set(4, 480)
    cam_2.set(3, 640)
    cam_2.set(4, 480)


def change_res(width, height):
    cam_1.set(3, width)
    cam_1.set(4, height)
    cam_2.set(3, width)
    cam_2.set(4, height)


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

#make_1080p()

# TODO brightness
#cam_1.set(10, value)
#cam_2.set(10, value)
#
## TODO contrast
#cam_1.set(11, value)
#cam_2.set(11, value)
#
## TODO saturatie
#cam_1.set(12, value)
#cam_2.set(12, value)
#
## TODO hue
#cam_1.set(13, value)
#cam_2.set(13, value)
#
## TODO gain
#cam_1.set(14, value)
#cam_2.set(14, value)
#
## TODO exposure
#cam_1.set(15, value)
#cam_2.set(15, value)

# TODO focus
# min: 0, max: 255, increment:5
#cam_1.set(28, 255)
#cam_2.set(28, 255)
print(cv2.CAP_PROP_FOCUS)
print(cv2.CAP_DSHOW)

print("create window")
cv2.namedWindow("Frame")
cv2.namedWindow("Frame2")
img_counter = 0

start = time.time()

while True:
    print("new")
    #ret_1, frame_1 = cam_1.read()
    #if not ret_1:
    #    print("failed to grab frame_1")
    #    break
    #frame_1 = rescale_frame(frame_1, 50)
    #cv2.imshow("Frame", frame_1)
    #cv2.waitKey(1)
    ret_2, frame_2 = cam_2.read()
    if not ret_2:
        print("failed to grab frame_2")
        break
    #frame_2 = rescale_frame(frame_2, 50)
    cv2.imshow("Frame2", frame_2)

    #both = np.concatenate((frame_1, frame_2), axis=1)
    #cv2.imshow('Frame', frame_1)
    cv2.waitKey(1)

    #if time.time() - start > 5:
    #    start = time.time()
    #
    #    img_name = "opencv_frame_{}.png".format(img_counter)
    #    cv2.imwrite(img_name, frame_1)
    #    print("{} written!".format(img_name))
    #    img_counter += 1
    #
    #    img_name_2 = "opencv_frame_{}.png".format(img_counter)
    #    cv2.imwrite(img_name_2, frame_2)
    #    print("{} written!".format(img_name_2))
    #    img_counter += 1

#cam_1.release()
cam_2.release()

cv2.destroyAllWindows()