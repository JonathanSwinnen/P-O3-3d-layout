import cv2
import math
import numpy as np
import pickle
import os

from numpy.core.fromnumeric import size


def calculate(camera_1, camera_2):
    """Calibrate the cameras
    Args:
        k: scale down factor
    returns:
        fov: field of view of the camera (diagonally)
        dir_1: direction of camera_1
        dir_2: direction of camera_2
        coord_1: coordinates of camera 1 (camera 1 is directly above the axis origin) -> coord_1 = np.array([0,0,height])
        coord_2: coordinates of camera 2 -> coord_2 = np.array([distance between cameras, 0, height])
    """

    scale_down = int(input("enter a scaledown factor:\n"))

    #   getting the frame size:
    ret_cal_1, frame_cal_1 = camera_1.read()
    if not ret_cal_1:
        print("failed to grab frame_1")
    frame_shape = frame_cal_1.shape
    image_size = (frame_shape[1] // scale_down, frame_shape[0] // scale_down)

    print(
        "____________________________________________________________________________________________CAMERA_POSITIONS______________"
    )
    fov_horizontal = float(
        input("what's the horizontal fov of the camera? 68° I think? (ask Daan):\n")
    )
    height = float(input("what's the height of both cameras?:\n"))
    width = float(input("what's the distance between both cameras?:\n"))

    first_time = 1
    while True:
        #   SHOW POINTS FOR DIRECTIONS OF CAMERAS
        ret_cal_1, frame_cal_1 = camera_1.read()
        if not ret_cal_1:
            print("failed to grab frame_1")
            break
        ret_cal_2, frame_cal_2 = camera_2.read()
        if not ret_cal_2:
            print("failed to grab frame_2")
            break
        if first_time:
            first_time = 0
            cv2.namedWindow("Calibration one...", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Calibration two...", cv2.WINDOW_NORMAL)
            cv2.imshow("Calibration one...", frame_cal_1)
            cv2.imshow("Calibration two...", frame_cal_2)
            cv2.waitKey(0)
            print(
                "__________look at the images. press any key._______________________________________________"
            )

            print(
                "____________________________________________________________________________________________EXPLANATION______________"
            )
            print(
                "The middle of both camera images will be visible (red squares), put an object in everyone of these two points"
            )
            print(
                "and measure their horizontal distance from the points to the cameras ('y' coord)"
            )
            print("also measure the 'x' coordinate of both points")
            print("Press space bar to stop the live feed and enter the values")

        #   making the center visible:
        cv2.rectangle(
            frame_cal_1,
            ((image_size[0] // 2) - 2, (image_size[1] // 2) - 2),
            ((image_size[0] // 2) + 2, (image_size[1] // 2) + 2),
            (0, 0, 253),
            2,
        )
        cv2.rectangle(
            frame_cal_2,
            (image_size[0] // 2 - 2, image_size[1] // 2 - 2),
            (image_size[0] // 2 + 2, image_size[1] // 2 + 2),
            (0, 0, 253),
            2,
        )

        cv2.imshow("Calibration one...", frame_cal_1)
        cv2.imshow("Calibration two...", frame_cal_2)
        key = cv2.waitKey(1)
        if key % 256 == 32:
            #   space pressed
            break
    print(
        "____________________________________________________________________________________________ENTER_NUMBERS:______________"
    )
    distance_center_point_camera_1 = float(
        input(
            "what's the horizontal distance of the center point of camera 1? ('y' coord):\n"
        )
    )
    x_point_1 = float(
        input("what's the 'x' coordinate of the center point of camera 1?:\n")
    )
    distance_center_point_camera_2 = float(
        input(
            "what's the horizontal distance of the center point of camera 2? ('y' coord):\n"
        )
    )
    x_point_2 = float(
        input("what's the 'x' coordinate of the center point of camera 2?:\n")
    )
    dir_1 = np.array([x_point_1, distance_center_point_camera_1, -height])
    dir_2 = np.array([x_point_2 - width, distance_center_point_camera_2, -height])
    rg1_norm = np.linalg.norm(dir_1)
    rg2_norm = np.linalg.norm(dir_2)
    dir_1= dir_1 / rg1_norm
    dir_2= dir_2/ rg2_norm
    coord_1 = np.array([0, 0, height])/100
    coord_2 = np.array([width, 0, height])/100
    cv2.destroyAllWindows()

    print("Calculating directional unit vectors")
    #   directions of the axis of the image on this plane:
    #   we want to create "normalized" vectors that follow both the axis of the image on this projection plane
    #           "normalized": their length is the projected length of 1 pixel on this plan
    #   x1 is horizontal, lateral to self.calibration_values["dir_1"]
    x1 = np.cross(np.array([0, 0, 1]), dir_1)
    #   we say x1's direction to have a positive x-value
    if x1[0] < 0:
        x1 = -x1

    fov_horizontal_rad = (fov_horizontal / 180) * math.pi

    #   calculation of the size of a pixel:
    d=0.5
    size_pixel = (2*d*math.tan(fov_horizontal_rad/2))/(image_size[0])

    # TODO : VERDER FIXEN VANAF HIER

    #   normalize x1 to be the same size as a pixel
    x1_norm = np.linalg.norm(x1)
    x1 = x1 / x1_norm  #   x is 1m long
    x1 = x1 * size_pixel  #    x is 1 pixel long
    
    #   y1 is lateral to self.calibration_values["dir_1"] and in a vertical plane
    #       this vertical plane has self.calibration_values["dir_1"] and the vertical vector in it:
    vertical_rg = np.cross(dir_1, np.array([0, 0, 1]))
    y1 = np.cross(vertical_rg, dir_1)
    #   we say y1's direction to have a negative y-value
    if y1[1] > 0:
        y1 = -y1
    #   normalize y1 as the size of one pixel
    y1_norm = np.linalg.norm(y1)
    y1 = y1 / y1_norm  #   is nu 1m lang
    y1 = y1 * size_pixel  #    is nu 1 pixel lang

    x2 = np.cross(np.array([0, 0, 1]), dir_2)
    #   x2's x-coordinate is positive:
    if x2[0] < 0:
        x2 = -x2
    x2_norm = np.linalg.norm(x2)
    x2 = x2 / x2_norm  #   now it is 1m long
    #   now let's find y2
    #       the plane made by C2M2 and (0,0,1) intersects bv2 allong y2
    verticaal_rg_2 = np.cross(dir_2, np.array([0, 0, 1]))
    y2 = np.cross(verticaal_rg_2, dir_2)
    if y2[2] > 0:
        y2 = -y2
    y2_norm = np.linalg.norm(y2)
    y2 = y2 / y2_norm  #   now it is 1m long
    x2 = x2 * size_pixel
    y2 = y2 * size_pixel

    calculated_dict = {
        "dir_1": dir_1,
        "dir_2": dir_2,
        "coord_1": coord_1,
        "coord_2": coord_2,
        "image_size": image_size,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }

    return calculated_dict


def save_calibration(filename, calculated):
    file = open(filename, "wb")
    pickle.dump(calculated, file)
    file.close()


def load_calibration(filename):
    print("1")
    file = open(filename, "rb")
    print("2")
    calculated = pickle.load(file)
    print("3")
    file.close()
    return calculated


