import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
imgL = cv.imread('one.jpg',0)
imgR = cv.imread('two.jpg',0)

scale_percent = 30  # percent of original size
width = int(imgL.shape[1] * scale_percent / 100)
height = int(imgL.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resizedL = cv.resize(imgL, dim, interpolation=cv.INTER_AREA)
resizedR = cv.resize(imgR, dim, interpolation=cv.INTER_AREA)

stereo = cv.StereoBM_create(numDisparities=256, blockSize=13)
disparity = stereo.compute(resizedL, resizedR)

f = 0.026;
d = 0.1;


plt.imshow(disparity,'gray')
plt.show()


