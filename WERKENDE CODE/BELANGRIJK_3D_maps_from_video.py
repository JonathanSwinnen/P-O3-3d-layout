import cv2 as cv
import numpy as np
import time
import pickle
from matplotlib import pyplot as plt
# timer
start = time.time()
step = 5


# defining vars
# defining vars
window_size = 3
min_disp = 16
num_disp = 112 - min_disp
"""
stereo = cv.StereoSGBM_create(minDisparity=0,
                              numDisparities=96,
                              blockSize=20,
                              P1=800,
                              P2=2000,
                              disp12MaxDiff=1,
                              uniquenessRatio=5,
                              speckleWindowSize=100,
                              speckleRange=32
                              )
"""
window_size = 5
min_disp = 16
num_disp = 112 - min_disp
stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                              numDisparities=num_disp,
                              blockSize=8,
                              P1=8 * 3 * window_size ** 2,
                              P2=32 * 3 * window_size ** 2,
                              disp12MaxDiff=1,
                              uniquenessRatio=10,
                              speckleWindowSize=100,
                              speckleRange=32
                              )
h = None

video_L = cv.VideoCapture('output_L.avi')
video_R = cv.VideoCapture('output_R.avi')

total_frames = int(video_L.get(cv.CAP_PROP_FRAME_COUNT))

success_L, success_R = True, True
count = 0
perc = 0
data = dict()
print(perc, "%")

while success_L and success_R:
    # read a frame
    success_L, imgL = video_L.read()
    success_R, imgR = video_R.read()

    # escape when out of frames
    if not(success_L and success_R):
        break



    # rectify images
    REMAP_INTERPOLATION = cv.INTER_LINEAR
    DEPTH_VISUALIZATION_SCALE = 2048

    calibration = np.load("out/calib_out.npz", allow_pickle=False)
    imageSize = tuple(calibration["imageSize"])
    leftMapX = calibration["leftMapX"]
    leftMapY = calibration["leftMapY"]
    leftROI = tuple(calibration["leftROI"])
    rightMapX = calibration["rightMapX"]
    rightMapY = calibration["rightMapY"]
    rightROI = tuple(calibration["rightROI"])

    fixedLeft = cv.pyrDown(cv.remap(imgL, leftMapX, leftMapY, REMAP_INTERPOLATION))
    fixedRight = cv.pyrDown(cv.remap(imgR, rightMapX, rightMapY, REMAP_INTERPOLATION))


    grayLeft = cv.cvtColor(fixedLeft, cv.COLOR_BGR2GRAY)
    grayRight = cv.cvtColor(fixedRight, cv.COLOR_BGR2GRAY)

    # define vars
    if h is None:
        h, w = imgL.shape[:2]
        f = 0.6 * w  # guess for focal length

        Q = np.float32([[1, 0, 0, -0.5 * w],
                        [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                        [0, 0, 0, -f],  # so that y-axis looks up
                        [0, 0, 1, 0]])

    # TODO: rectify images!!!!
    # calculate disparity map
    disp = stereo.compute(grayRight, grayLeft).astype(np.float32) / 16.0
    if count == 10:
        plt.imshow(disp, "gray")
        plt.show()

    # calculate depth map
    points = cv.reprojectImageTo3D(disp, Q)

    colors = cv.cvtColor(fixedLeft, cv.COLOR_BGR2RGB)  # colorizing 3D map

    mask = disp > disp.min()+1
    points = points[mask]
    colors = colors[mask]

    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    print(points)


    data[count] = (points, colors)
    count += 1

    if (perc + step)/100 < count/total_frames:
        perc += step
        print(perc, "%")

print("duration: ", time.time() - start)
print("frames processed:", count)

with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(data, f)
