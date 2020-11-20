import numpy as np
import cv2
from VideoPeopleMonitor import VideoPeopleMonitor
import os

# bestand locaties
dirname = os.path.dirname(__file__)
vid_path_1 = os.path.join(dirname, "data/videos/output_two_person_0.avi")
vid_path_2 = os.path.join(dirname, "data/videos/output_two_person_1.avi")
timestamps_path = os.path.join(
    dirname, "data/timestamps/output_two_person_timestamps.txt"
)
calib_path = os.path.join(dirname, "data/calib.pckl")

vm = VideoPeopleMonitor(calib_path, vid_path_2, vid_path_1, timestamps_path)
cv2.namedWindow("Camera one...")
cv2.namedWindow("Camera two...")

while True:
    frame_1, frame_2, _ = vm.get_frames()
    data, boxes = vm.update()

    for pers in data:
        tracked_points = { pers : data[0] for (pers, data) in data.items() }

        point_on_img = (data[pers][1], data[pers][2])
        # print on image 1(point_on_img)
        a, b, c, d = int(point_on_img[0][0]), int(point_on_img[0][1]), 20, 20
        cv2.rectangle(frame_1, (a - c, b - d), (a + c, b + d), (255, 0, 210), 5)
        cv2.putText(frame_1, pers, (a, b), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 255)

        # print on image 2(point_on_img)
        a, b, c, d = int(point_on_img[1][0]), int(point_on_img[1][1]), 20, 20
        cv2.rectangle(frame_2, (a - c, b - d), (a + c, b + d), (255, 0, 210), 5)
        cv2.putText(frame_2, pers, (a, b), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 255)

    # Plot boxes
    i = 0
    for box in boxes[0]:
        cv2.rectangle(frame_1, (box[0], box[1]), (box[2], box[3]), (50, 50, 200))
        cv2.putText(
            frame_1,
            str(i),
            (box[0], box[1]),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            255,
        )
        i += 1
    i = 0
    for box in boxes[1]:
        cv2.putText(
            frame_2,
            str(i),
            (box[0], box[1]),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            255,
        )
        cv2.rectangle(frame_2, (box[0], box[1]), (box[2], box[3]), (50, 50, 200))
        i += 1
    # cv2.waitKey()
    # print("\nFRAME\n")
    cv2.imshow("Camera one...", frame_1)
    cv2.imshow("Camera two...", frame_2)
    cv2.waitKey(1)
    print("data:", data)
    print()
    
