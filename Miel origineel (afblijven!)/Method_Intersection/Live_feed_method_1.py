import cv2
from Location_Detection import XYZ_Point
import Recognition # type: ignore
import numpy as np

#   This file tries to combine everything in one:
#       - image capturing
#       - person recognition
#       - location determination
#       - distance calculation

#   CALIBRATION
(
    fov,
    dir_1,
    dir_2,
    coord_1,
    coord_2,
    camera_1,
    camera_2,
) = Recognition.Camera_calibration(3)

#   IMAGE CAPTURING, select the desired number of cameras to be used (1 and 2) right here:
cv2.namedWindow("Recognition one...")
cv2.namedWindow("Recognition two...")

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    #   ret_1 is to check if a fail occurred,
    #   frame is the captured camera image
    ret_1, frame_1 = camera_1.read()
    if not ret_1:
        print("failed to grab frame_1")
        break

    ret_2, frame_2 = camera_2.read()
    if not ret_2:
        print("failed to grab frame_2")
        break

    # cv2.imshow("test", frame)
    coordinates_1 = Recognition.Recognize(frame_1, 1)
    coordinates_2 = Recognition.Recognize(frame_2, 2)
    # coordinates_1 = Recognition.Recognize_2(frame_1, 1)
    # coordinates_2 = Recognition.Recognize_2(frame_2, 2)
    key = cv2.waitKey(1)
    if key % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        cv2.destroyAllWindows()
        break
    elif key % 256 == 32:
        #   space pressed:
        #       pause right now, changes being made to code
        print("paused, press space to continue")
        while True:
            key = cv2.waitKey(1)
            if key % 256 == 32:
                #   play
                print("play")
                break
    elif not (coordinates_2 == [] or coordinates_1 == []):
        # both camera's see a face
        # this frame needs to be saved and calculated!
        # TODO: recognition (Mathias en Mathias): find the best haarcascade and
        # for now:
        image_size = np.array([frame_1.shape[1], frame_1.shape[0]])
        XYZ = XYZ_Point( # type: ignore
            image_size,
            fov,
            coordinates_1,
            coordinates_2,
            coord_1,
            coord_2,
            dir_1,
            dir_2,
        ) 
        cv2.namedWindow("Position Map", cv2.WINDOW_NORMAL)
        #   MAYBE YOU'll HAVE TO INCLUDE A FULL path HERE, like so:
        #   grid = cv2.imread(r"C:\Users\peete\Documents\Burgi\2020-2021\P&O3\Grid_P_O_3.jpg")
        #           (but then your exact path)
        grid = cv2.imread("Grid_P_O_3.jpg")
        for Point in XYZ:
            cv2.circle(
                grid,
                (
                    int(Point[0] * 200),
                    int(1000 - (Point[1] * 200)),
                ),
                10,
                (0, 0, 255),
                -1,
            )
            # cv2.putText(grid,Point[2],(int(Point[0]*200 + 20),int(1000 - (Point[1]*200))), font, 4,(255,255,255),2)
        cv2.resizeWindow("Position Map", 1000, 1000)
        cv2.imshow("Position Map", grid)

        print(XYZ)