import cv2
from Positioner import Positioner
from VideoPeopleMonitor import VideoPeopleMonitor
import os
import numpy as np

# bestand locaties
dirname = os.path.dirname(__file__)
vid_path_1 = os.path.join(dirname, "data/videos/output_more_person_0.avi")
vid_path_2 = os.path.join(dirname, "data/videos/output_more_person_1.avi")
timestamps_path = os.path.join(
    dirname, "data/timestamps/meerdere_pers.txt"
)
calib_path = os.path.join(dirname, "data/calib.pckl")
boxes_path = os.path.join(dirname, "data/video_data/meerdere_pers.pckl")

vm = VideoPeopleMonitor(calib_path, vid_path_2, vid_path_1, timestamps_path, boxes_path)
cv2.namedWindow("Camera one...",cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera two...",cv2.WINDOW_NORMAL)
COLOR = {"Daan":(95, 168, 199),"Jonathan":(99, 199, 99),"Mathias P.":(242, 162, 104), "Mathias S.":(200,200,200), "Ebert":(200,40,50), "Miel":(43,0,30)}

while True:
    cv2.waitKey()
    print("\nFRAME " + str(vm.frame_count) + "\n")
    frame_1, frame_2, _ = vm.get_frames()
    data, boxes, pred, dets, coords = vm.update()

    bd = 40
    frame_1 = cv2.copyMakeBorder(frame_1,bd,bd,bd,bd,cv2.BORDER_CONSTANT,value=(255,255,255))
    frame_2 = cv2.copyMakeBorder(frame_2,bd,bd,bd,bd,cv2.BORDER_CONSTANT,value=(255,255,255))
    for pers in data:
        #tracked_points = { pers : data[0] for (pers, data) in data.items() }

        predpoint = pred[pers][0]
        print("repb", pred[pers])
        point_on_img = vm.positioner.reprojectPoint(predpoint)
        # print on image 1(point_on_img)
        a, b, c, d = int(point_on_img[0][0])+bd, int(point_on_img[0][1])+bd, 20, 20
        cv2.rectangle(frame_1, (a - c, b - d), (a + c, b + d), (100,100,100), 2)

        # print on image 2(point_on_img)
        a, b, c, d = int(point_on_img[1][0])+bd, int(point_on_img[1][1])+bd, 20, 20
        cv2.rectangle(frame_2, (a - c, b - d), (a + c, b + d), (100,100,100), 2)

        point_on_img = (data[pers][1], data[pers][2])
        # print on image 1(point_on_img)
        a, b, c, d = int(point_on_img[0][0])+bd, int(point_on_img[0][1])+bd, 20, 20
        cv2.rectangle(frame_1, (a - c, b - d), (a + c, b + d), COLOR[pers], 2)
        cv2.putText(frame_1, pers, (a, b-25), cv2.FONT_HERSHEY_DUPLEX , 0.6, COLOR[pers])

        # print on image 2(point_on_img)
        a, b, c, d = int(point_on_img[1][0])+bd, int(point_on_img[1][1])+bd, 20, 20
        cv2.rectangle(frame_2, (a - c, b - d), (a + c, b + d), COLOR[pers], 2)
        cv2.putText(frame_2, pers, (a, b-25), cv2.FONT_HERSHEY_DUPLEX , 0.6, COLOR[pers])
    if dets is None: dets = []

    for c in coords[0]:
        a, b, c, d = c[0]+bd, c[1]+bd, 3, 3
        cv2.rectangle(frame_1, (a - c, b - d), (a + c, b + d), (0,0,255), 3)
    for c in coords[1]:
        a, b, c, d = c[0]+bd, c[1]+bd, 3, 3
        cv2.rectangle(frame_2, (a - c, b - d), (a + c, b + d), (0,0,255), 3)

    k = 0
    for det in dets:
        point_on_img = vm.positioner.reprojectPoint(np.array([det]).T)
        a, b, c, d = int(point_on_img[0][0])+bd-3, int(point_on_img[1][1])+bd-3, 3, 3
        cv2.rectangle(frame_1, (a - c, b - d), (a + c, b + d), (255*(k/len(dets)),255*(1-k/len(dets)),0), 3)
        a, b, c, d = int(point_on_img[1][0])+bd-3, int(point_on_img[1][1])+bd-3, 3, 3
        cv2.rectangle(frame_2, (a - c, b - d), (a + c, b + d), (255*(k/len(dets)),255*(1-k/len(dets)),0), 3)
        k += 1

    # Plot boxes
    i = 0
    for box in boxes[0]:
        cv2.rectangle(frame_1, (box[0]+bd, box[1]+bd), (box[2]+bd, box[3]+bd), (199, 199, 199),1)
        cv2.putText(
            frame_1,
            str(i),
            (box[0] + bd + 5, box[1]+ bd +20),
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
            (box[0]+5+bd, box[1]+20+bd),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            255,
        )
        cv2.rectangle(frame_2, (box[0]+bd, box[1]+bd), (box[2]+bd, box[3]+bd), (199, 199, 199),1)
        i += 1
    # cv2.waitKey()
    print("\nFRAME\n")
    cv2.imshow("Camera one...", frame_1)
    cv2.imshow("Camera two...", frame_2)
    #cv2.waitKey()
    cv2.waitKey(100)
    print("data:", data)
    print()
    
