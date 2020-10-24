#!/usr/bin/env python

# Gebaseerd op: https://github.com/opencv/opencv/blob/3.4.1/samples/python/stereo_match.py

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

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


if __name__ == '__main__':
    print('loading images...')
    imgL = cv.pyrDown(cv.imread('L0.png'))  # downscale images for faster processing
    imgR = cv.pyrDown(cv.imread('R0.png'))

    # disparity range is tuned for 'aloe' image pair
    window_size = 7
    min_disp = 0
    num_disp = 112-min_disp
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 2,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 0,
        uniquenessRatio = 15,
        speckleWindowSize = 100,
        speckleRange = 32
    )

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

    print('computing disparity...')
    disp = stereo.compute(fixedLeft, fixedRight).astype(np.float32) / 16.0

    cv.imshow('left', fixedLeft)
    cv.imshow('right', fixedRight)
    cv.imshow('disparity', (disp-min_disp)/num_disp)

    rstereo = cv.ximgproc.createRightMatcher(stereo)
    dispr = rstereo.compute(fixedRight, fixedLeft).astype(np.float32) / 16.0
    print(dispr)
    wls_filter = cv.ximgproc.createDisparityWLSFilter(stereo);
    wls_filter.setLambda(8000);
    wls_filter.setSigmaColor(0.8);
    dispf = wls_filter.filter(disp, fixedLeft, disparity_map_right=dispr);

    print('generating 3d point cloud...',)
    h, w = fixedLeft.shape[:2]
    f = 0.6*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv.reprojectImageTo3D(disp, Q)
    colors = cv.cvtColor(fixedLeft, cv.COLOR_BGR2RGB)
    mask = disp > disp.min()+1
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print('%s saved' % 'out.ply')


    cv.imshow('disparity filtered', (dispf-min_disp)/num_disp)
    cv.waitKey()
    cv.destroyAllWindows()
