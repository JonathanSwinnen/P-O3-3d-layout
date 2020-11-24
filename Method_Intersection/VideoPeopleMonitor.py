import Calibration
from Detector import Detector
from Positioner import *
from Tracker import Tracker
from time import perf_counter
import os
import cv2
import matplotlib.pyplot as plt
from math import floor
import numpy as np


class VideoPeopleMonitor:
    def __init__(self, calib_path, vid_path_1, vid_path_2, person_timestamps_path):

        f = open(person_timestamps_path, "r")
        lines = f.read().split("\n")
        self.timestamps = {}
        for line in lines:
            split = line.split(",")
            self.timestamps[int(split[0])] = split[1:]
        self.camera_1 = cv2.VideoCapture(vid_path_1)
        self.camera_2 = cv2.VideoCapture(vid_path_2)

        self.dt = 1/self.camera_1.get(cv2.CAP_PROP_FPS)

        self.calibrated_values = Calibration.load_calibration(calib_path)
        self.image_size = self.calibrated_values["image_size"]

        self.detector = Detector()

        #   Initialize the tracker with appropriate values:
        u = 0 * np.ones((3, 1))
        stac = 0.95
        stdm = np.array([[0.1], [0.1], [0.9]])
        self.tracker = Tracker(u, stac, stdm, 0.1, 6, 0.3, 0.1)

        self.positioner = Positioner(
            self.calibrated_values, 0.002, 0.2, ([-1, 0, 0], [6, 7, 3])
        )

        self.frame_1 = None
        self.frame_2 = None
        self.has_captured_frame = False

        self.frame_count = 1890
        self.camera_1.set(cv2.CAP_PROP_POS_FRAMES,self.frame_count)
        self.camera_2.set(cv2.CAP_PROP_POS_FRAMES,self.frame_count)

        ts_file = open(person_timestamps_path, "r")
        ts_lines = ts_file.read().split("\n")
        self.timestamps = {}
        for line in ts_lines:
            split = line.split(",")
            self.timestamps[int(split[0])] = split[1:]

    def get_frames(self):
        self.has_captured_frame = False
        ret_1, frame_1 = self.camera_1.read()
        if not ret_1:
            print("failed to grab frame_1")
            return None, None, False

        ret_2, frame_2 = self.camera_2.read()
        if not ret_2:
            print("failed to grab frame_2")
            return None, None, False

        self.frame_1 = cv2.resize(frame_1, self.image_size)
        self.frame_2 = cv2.resize(frame_2, self.image_size)
        self.has_captured_frame = True
        self.frame_count += 1
        return self.frame_1, self.frame_2, True

    def update(self):
        if self.has_captured_frame:
            self.has_captured_frame = False
            #   checks the timestamps file and adds/removes a person if necessary
            event = self.timestamps.get(self.frame_count, None)
            print("current frame count:", self.frame_count)
            if event is not None:
                if event[1] == "Exit":
                    self.tracker.rm_person(event[0])
                elif event[1] == "Enter":
                    if event[2] == "R":
                        self.tracker.add_person(
                            event[0], np.array([[4], [4.5], [1.3], [0], [0], [0]]))
                    elif event[2] == "L":
                        self.tracker.add_person(
                            event[0], np.array([[0], [5], [1.3], [0], [0], [0]]))
            # make tracker prediction
            prediction = self.tracker.predict(self.dt)

            #   recognize every person in every frame:
            (
                coordinates_1,
                coordinates_2,
                boxes_1,
                boxes_2,
            ) = self.detector.detect_both_frames(self.frame_1, self.frame_2)

            # this frame needs to be saved and calculated!
            # detect points
            dets = self.positioner.get_XYZ(coordinates_1, coordinates_2, prediction)
            # update filter
            if dets:
                print("debug")
            tracked_points = self.tracker.update(dets)

            data_dict = {}

            for pers in tracked_points:
                point_on_img = self.positioner.reprojectPoint(tracked_points[pers][0])
                data_dict[pers] = (
                    tracked_points[pers][0],
                    point_on_img[0],
                    point_on_img[1],
                )

            return data_dict, (boxes_1, boxes_2)

        else:
            return None
