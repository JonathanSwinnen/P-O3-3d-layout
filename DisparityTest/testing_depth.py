import numpy as np
import cv2 as cv
#from matplotlib import pyplot as plt
#
#imgL = cv.imread('one.jpg', 0)
#imgR = cv.imread('two.jpg', 0)
#
#scale_percent = 50  # percent of original size
#width = int(imgL.shape[1] * scale_percent / 100)
#height = int(imgL.shape[0] * scale_percent / 100)
#dim = (width, height)
## resize image
#resizedL = cv.resize(imgL, dim, interpolation=cv.INTER_AREA)
#resizedR = cv.resize(imgR, dim, interpolation=cv.INTER_AREA)
#
#stereo = cv.StereoBM_create(numDisparities=256, blockSize=15)
#disparity = stereo.compute(resizedL, resizedR)
#
#f = 0.026
#d = 0.1
#
#print(type(disparity))
#
#
#plt.imshow(disparity,'gray')
#plt.show()

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

print('loading images...')
imgL = cv.pyrDown(cv.imread(cv.samples.findFile('opencv_0.png')))  # downscale images for faster processing
imgR = cv.pyrDown(cv.imread(cv.samples.findFile('opencv_1.png')))

im1 = [
  [663.42471377, 0., 300.16117822],
  [0., 663.26095825, 235.76890967],
  [0., 0., 1.]
]

d1 = [
  [-3.67872499e-02 - 3.73117624e-01, 2.20644582e-03 - 4.88829930e-04,
    1.17858286e+00
  ]
]

im2 = [
  [656.93710911, 0., 311.25326654],
  [0., 656.51398717, 238.7629016],
  [0., 0., 1.]
]
d2 = [
  [1.93171875e-02, -9.63240763e-01, -1.01823352e-03, -2.45940100e-03
    2.95730921e+00
  ]
]
r = [
  [0.99950681, -0.01853554, -0.02534887]
  [0.01956959, 0.9989604, 0.04117227]
  [0.02455937, -0.04164803, 0.99883046]
]
t = [
  [-5.11936755],
  [-0.16800592],
  [-0.07269131]
]

e = [
  [-2.70357993e-03, 7.96128605e-02, -1.64816567e-01],
  [5.30729729e-02, -2.11864197e-01, 5.11522287e+00],
  [6.77391179e-02, -5.11715952e+00, -2.15034733e-01]
]

f = [
  [2.88865988e-08, -8.50839483e-07, 1.36021829e-03],
  [-5.67427556e-07, 2.26569679e-06, -3.66460277e-02],
  [-3.48977559e-04, 3.56505482e-02, 1.00000000e+00]
]



stereoRectify(im1, d1, im2, d2, Size(640, 480), r, t, R1, R2, P1, P2, Q,
    CALIB_ZERO_DISPARITY, 1, Size(320, 240), &validRoI[0], &validRoI[1]);



# disparity range is tuned for 'aloe' image pair
window_size = 3
min_disp = 16
num_disp = 112 - min_disp
stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                              numDisparities=num_disp,
                              blockSize=16,
                              P1=8 * 3 * window_size ** 2,
                              P2=32 * 3 * window_size ** 2,
                              disp12MaxDiff=1,
                              uniquenessRatio=10,
                              speckleWindowSize=100,
                              speckleRange=32
                              )

print('computing disparity...')
disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

print('generating 3d point cloud...', )
h, w = imgL.shape[:2]
f = 0.8 * w  # guess for focal length
Q = np.float32([[1, 0, 0, -0.5 * w],
                [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                [0, 0, 0, -f],  # so that y-axis looks up
                [0, 0, 1, 0]])
points = cv.reprojectImageTo3D(disp, Q)
colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
mask = disp > disp.min()
out_points = points[mask]
out_colors = colors[mask]
out_fn = 'out.ply'
write_ply(out_fn, out_points, out_colors)
print('%s saved' % out_fn)

cv.imshow('left', imgL)
cv.imshow('disparity', (disp - min_disp) / num_disp)
cv.waitKey()

print('Done')


