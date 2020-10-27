import cv2 as cv
import numpy as np
import time
from vedo import *
import pickle

with open('objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    data = pickle.load(f)

print(data[10][0])

print(1)
vp = Plotter(title="3D point cloud", axes=1, interactive=1)
print(2)
pts = Points(data[10][0], r=1, c=data[10][1])
print(3)
vp.add(pts)
print(4)
vp.show()
#first = True
#for i in range(len(data)):
#    pts = Points(data[i][0], r=1, c=data[i][1])
#    if not first:
#        vp.clear()
#    else:
#        first = False
#    vp.add(pts)
#    vp.show(axes=1)
#    time.sleep(1)

vp.show(interactive=1, axes=1)