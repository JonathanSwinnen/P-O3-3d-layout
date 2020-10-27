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
import time
import math
import copy

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


def norm(v1, v2):
    """
    :return: distance between 2 vectors
    """
    return math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2 + (v1[2]-v2[2])**2)
    # return np.linalg.norm(np.subtract(v1, v2))


def points_to_check(i, j, filter):
    """
    :param i: pixel row
    :param j: pixel column
    :param filter: groups matrix
    :return: all adjacent pixels not yet in a group
    """
    to_get = []
    if i != 0 and filter[i-1][j] is None:
        to_get += [(i-1, j)]
    if i != len(filter)-1 and filter[i+1][j] is None:
        to_get += [(i+1, j)]
    if j != 0 and filter[i][j-1] is None:
        to_get += [(i, j-1)]
    if j != len(filter[i])-1 and filter[i][j+1] is None:
        to_get += [(i, j+1)]
    return to_get


def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


if __name__ == '__main__':
    start = time.time()
    print('loading images...')
    imgL = cv.imread('L2.png')  # downscale images for faster processing
    imgR = cv.imread('R2.png')

    # disparity range is tuned for 'aloe' image pair
    window_size = 7
    min_disp = 8
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

    #print('computing disparity...')
    disp = stereo.compute(fixedLeft, fixedRight).astype(np.float32) / 16.0

    #cv.imshow('left', fixedLeft)
    #cv.imshow('right', fixedRight)
    #cv.imshow('disparity', (disp-min_disp)/num_disp)

    rstereo = cv.ximgproc.createRightMatcher(stereo)
    dispr = rstereo.compute(fixedRight, fixedLeft).astype(np.float32) / 16.0
    #print(dispr)

    wls_filter = cv.ximgproc.createDisparityWLSFilter(stereo)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(2.5)
    dispf = wls_filter.filter(disp, fixedLeft, disparity_map_right=dispr)

    #print(dispf)
    #print('generating 3d point cloud...',)
    h, w = fixedLeft.shape[:2]
    f = 0.6 * w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv.reprojectImageTo3D(dispf, Q)
    colors = cv.cvtColor(fixedLeft, cv.COLOR_BGR2RGB)

    print("start filtering")

    # zieke hacks om 540 * 940 None's te krijgen
    # [None] * x gaat niet, want obj gekoppeld
    objects = []
    for i in range(len(points)):
        appending = []
        for j in range(len(points[0])):
            appending.append(None)
        objects.append(appending)

    # go search objects
    counter = 1
    for i in range(len(points)):
        # calc every pixels once
        for j in range(len(points[0])):
            # if no group yet
            if objects[i][j] is None:
                # add to current group
                objects[i][j] = counter
                to_check = [(i, j)]
                while len(to_check) > 0:
                    p1 = to_check.pop()
                    # get all adjacent pixels that don't have a group yet
                    for p2 in points_to_check(p1[0], p1[1], objects):
                        # if distance between points is less than 0.2
                        if norm(points[p1[0]][p1[1]], points[p2[0]][p2[1]]) < 0.2: # 0.2 can be adjusted!
                            # add to group and search adjacent pixels
                            objects[p2[0]][p2[1]] = counter
                            to_check += [p2]
                # group is closed, next group
                counter += 1

    # convert to numpy array, needed for mask
    for row in objects:
        row = np.array(row)
    objects = np.array(objects)

    mask = dispf > dispf.min()+1
    points = points[mask]
    colors = colors[mask]
    objects = objects[mask]

    # count amount of pixels in group
    numbers = {}
    for i in range(0, len(objects)):
        if objects[i] in numbers:
            numbers[objects[i]] += 1
        else:
            numbers[objects[i]] = 1

    # go search all groups with enough pixels
    list_good_nb = []
    for key, val in numbers.items():
        if val > 5000: # amount of pixels in group, can be adjusted
            list_good_nb += [key]
    print(len(numbers.keys()), len(list_good_nb))

    # export all good groups
    #for i in range(0, len(list_good_nb)):
    #    mask2 = objects == list_good_nb[i]
    #    out_points = points[mask2]
    #    out_colors = colors[mask2]
#
    #    out_fn = 'out_' + str(i) + '.ply'
    #    write_ply(out_fn, out_points, out_colors)
    #    print('%s saved' % 'out.ply')
#
    ##cv.imshow('disparity filtered', (dispf-min_disp)/num_disp)
    ##cv.waitKey()
    ##cv.destroyAllWindows()
    print(time.time()-start)
