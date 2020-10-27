import cv2
import numpy as np
from math import floor


def Camera_calibration(k):
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
    camera_1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    camera_2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    #   getting the full resolution:
    camera_1.set(cv2.CAP_PROP_FRAME_WIDTH, floor(1920 / k))
    camera_2.set(cv2.CAP_PROP_FRAME_WIDTH, floor(1920 / k))
    camera_1.set(cv2.CAP_PROP_FRAME_HEIGHT, floor(1080 / k))
    camera_2.set(cv2.CAP_PROP_FRAME_HEIGHT, floor(1080 / k))
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
            print(
                "__________look at the images. press any key._______________________________________________"
            )
            cv2.waitKey(0)
            reversed = int(input("are the cameras switched? (1:Yes, 0:No)\n"))
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

            if reversed:
                camera_1, camera_2 = camera_2, camera_1

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

    return fov_horizontal, dir_1, dir_2, coord_1, coord_2, camera_1, camera_2


def Recognize(image: np.ndarray, one_or_two):
    """recognizes a central point of a person in an image

    Args:
        image: the image where the person is in
        one_or_two: camera one or camera two

    returns:
        the point on the image
    """

    #   Select the wanted haarcascade path right here:
    cascPath = r"C:\Users\peete\Documents\Burgi\2020-2021\Daan_bestanden\detection_daan\haarcascades\haarcascade_frontalface_alt2.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=2, minSize=(10, 10)
    )

    # flags = cv2.CV_HAAR_SCALE_IMAGE)

    #   points (x,y)
    faces_coordinates = []
    # if not len(faces) == 0:
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        a = (w // 2) - 3
        b = (w // 2) + 3
        c = x + a
        d = y + a
        cv2.rectangle(image, (c, d), (c + 6, d + 6), (0, 100, 250), 2)
        faces_coordinates += [(x + w // 2, y + h // 2)]

    #   display in the right window
    if one_or_two == 1:
        cv2.imshow("Recognition one...", image)
    else:
        cv2.imshow("Recognition two...", image)

    return faces_coordinates


def Recognize_2(frame, one_or_two):

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    (regions, _) = hog.detectMultiScale(
        frame, winStride=(4, 4), padding=(0, 0), scale=1.2
    )
    #   points (x,y)
    faces_coordinates = []
    # if not len(faces) == 0:
    for (x, y, w, h) in regions:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        a = (w // 2) - 3
        b = (w // 2) + 3
        c = x + a
        d = y + a
        cv2.rectangle(frame, (c, d), (c + 6, d + 6), (0, 100, 250), 2)
        faces_coordinates += [(x + w // 2, y + h // 2)]
        #   display in the right window
    if one_or_two == 1:
        cv2.imshow("Recognition one...", frame)
    else:
        cv2.imshow("Recognition two...", frame)

    return faces_coordinates