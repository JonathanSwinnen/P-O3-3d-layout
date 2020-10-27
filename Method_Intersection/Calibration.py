import cv2
import numpy as np
import pickle


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
    #   getting the frame size:
    ret_cal_1, frame_cal_1 = camera_1.read()
    if not ret_cal_1:
        print("failed to grab frame_1")
    frame_shape = frame_cal_1.shape
    image_size = (frame_shape[1], frame_shape[0])

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
    coord_1 = np.array([0, 0, height])
    coord_2 = np.array([width, 0, height])
    cv2.destroyAllWindows()

    calculated_dict = {"fov_h":fov_horizontal,"dir_1":dir_1,"dir_2":dir_2,"coord_1":coord_1,"coord_2":coord_2,"image_size":image_size}
    return calculated_dict


def save_calibration(filename, calculated):
    file = open(filename, "wb")
    pickle.dump(calculated, file)
    file.close()


def load_calibration(filename):
    file = open(filename, "rb")
    calculated = pickle.load(file)
    file.close()
    return calculated
