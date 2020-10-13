"""deze code is grotendeels geript van github
TODO: eigen versie (her)schrijven
"""

import sys
import numpy as np
import cv2

REMAP_INTERPOLATION = cv2.INTER_LINEAR
DEPTH_VISUALIZATION_SCALE = 2048

calibration = np.load("out/calib_out.npz", allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


stereoMatcher = cv2.StereoBM_create()
stereoMatcher.setMinDisparity(1)
stereoMatcher.setNumDisparities(128)
stereoMatcher.setBlockSize(21)
stereoMatcher.setROI1(leftROI)
stereoMatcher.setROI2(rightROI)
stereoMatcher.setSpeckleRange(16)
stereoMatcher.setSpeckleWindowSize(45)

# Grab both frames first, then retrieve to minimize latency between cameras
while(True):
    """if not left.grab() or not right.grab():
        print("No more frames")
        break"""

    leftFrame = cv2.imread("pic/LEFT/opencv_frame_3.png")
    rightFrame = cv2.imread("pic/RIGHT/opencv_frame_2.png")
    cv2.imshow("img", leftFrame)

    fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
    fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)

    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    depth = stereoMatcher.compute(grayLeft, grayRight)

    cv2.imshow('left', fixedLeft)
    cv2.imshow('right', fixedRight)
    cv2.imshow('depth', depth / DEPTH_VISUALIZATION_SCALE)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()