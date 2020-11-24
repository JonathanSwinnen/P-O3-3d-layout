"""

This is a helper_code file that can be used to manually create timestamp files for person entry/exits from the room

Normally our plan would be to track who is coming in/out of the room by requiring a card scan before entry or exit, 
and by sending this data back to the tracker. However, since the Covid pandemic we cannot work in a physical room anymore 
and we have to work with pre-recorded videos. These videos don't have timestamps of card scans, so for a workaround we store
entries/exits in a text file.

Create a text file, and use this program to open two videos (edit vid_path variables). Press or hold a key to play frames of the video.
When a person enters or leaves the room, add a line to a text file with the following format:

<frame number>,<person name>,<Enter/Exit>,<R/L>

<frame number>: the frame where the person enters the room. This is printed in the terminal when a frame is displayed.
<person name>: the person name that gets sent to the tracker
<Enter/Exit>: "Enter" when the person enters, "Exit" when the person exits
<R/L>: "R" when the person goes through the entry/exit on the right side, "L" when the person goes through the entry/exit on the left side

"""

import os
import cv2
import numpy as np

vid_1_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', 'videos', 'output_two_person_0.avi'))
vid_2_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', 'videos', 'output_two_person_1.avi'))

cam1 = cv2.VideoCapture(vid_1_path)
cam2 = cv2.VideoCapture(vid_2_path)


n = 0

cv2.namedWindow("1")
cv2.namedWindow("2")

while True:
    n += 1

    _, frame1 = cam1.read()
    _, frame2 = cam2.read() 

    cv2.imshow("1", frame1)
    cv2.imshow("2", frame2)

    print(n)

    cv2.waitKey()



