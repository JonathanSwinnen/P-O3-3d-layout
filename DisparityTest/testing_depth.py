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


