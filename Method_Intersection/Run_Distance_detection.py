import Calibration
from Detector import Detector
from Positioner import *
import cv2
from math import floor

def camera_setup():

    k = int(input("enter a scaledown factor:\n"))
    cv2.namedWindow("Camera one...")
    cv2.namedWindow("Camera two...")
    camera_1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    camera_2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    #   getting the right resolution:
    camera_1.set(cv2.CAP_PROP_FRAME_WIDTH, floor(1920 / k))
    camera_2.set(cv2.CAP_PROP_FRAME_WIDTH, floor(1920 / k))
    camera_1.set(cv2.CAP_PROP_FRAME_HEIGHT, floor(1080 / k))
    camera_2.set(cv2.CAP_PROP_FRAME_HEIGHT, floor(1080 / k))
    ret_cal_1, frame_cal_1 = camera_1.read()
    if not ret_cal_1:
        print("failed to grab frame_1")
    ret_cal_2, frame_cal_2 = camera_2.read()
    if not ret_cal_2:
        print("failed to grab frame_1")
    cv2.imshow("Camera one...", frame_cal_1)
    cv2.imshow("Camera two...", frame_cal_2)
    #   are the cameras reversed?
    reversed = bool(
        int(input("Enter 1 if the cameras are reversed. Enter 0 if they are right"))
    )
    if reversed:
        camera_1, camera_2 = camera_2, camera_1
    return camera_1, camera_2

def calibrate_cameras():
    calibrate = int(input("Do you want to calibrate the camera? (1:Yes, 0:No):\n"))

    if calibrate:
        #   CALIBRATION
        #   calibrated_values = (fov, dir_1, dir_2, coord_1, coord_2)
        calibrated_values = Calibration.calculate(camera_1, camera_2)

        Calibration.save_calibration(FILENAME, calibrated_values)
    else:
        calibrated_values = Calibration.load_calibration(FILENAME)

    return calibrated_values

def get_frames(camera_1,camera_2):
    ret_1, frame_1 = camera_1.read()
    if not ret_1:
        print("failed to grab frame_1")
        return None

    ret_2, frame_2 = camera_2.read()
    if not ret_2:
        print("failed to grab frame_2")
        return None 

    return frame_1, frame_2


FILENAME = r"data\calib.pckl"
detector = Detector()

camera_1 , camera_2 = camera_setup()
calibrated_values = calibrate_cameras()

positioner = Positioner(calibrated_values)

# TODO: GUI

while True:
    #   This loop embodies the main workings of this method
    frame_1, frame_2 = get_frames(camera_1, camera_2)
    
    if  frame_1 and frame_2:
            
        #   recognize every person in every frame:
        coordinates_1, coordinates_2 = detector.detect_both_frames(frame_1, frame_2)
        
        key = cv2.waitKey(1)

        #TODO: replace with GUI functions
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
        else:
            # both cameras recognize something
            # this frame needs to be saved and calculated!
            
            XYZ = positioner.get_XYZ(coordinates_1,coordinates_2)
    else:
        print("Frame skipped.")


#TODO: display this in the GUI
# for (x, y, w, h) in faces:
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             a = (w // 2) - 3
#             b = (w // 2) + 3
#             c = x + a
#             d = y + a
#             cv2.rectangle(image, (c, d), (c + 6, d + 6), (0, 100, 250), 2)
#         #   display in the right window
#         if one_or_two == 1:
#             cv2.imshow("Recognition one...", image)
#         else:
#             cv2.imshow("Recognition two...", image)
