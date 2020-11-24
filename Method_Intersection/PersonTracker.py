"""
This file contains a class containing the logic for a person tracker using a Kalman filter.

#TODO: expand algorithm

"""

from KalmanFilter import KalmanFilter
import numpy as np
from math import log

class KalmanPersonTracker:

    def __init__(self, id, x0, u, std_acc, std_meas, dt, max_certain_speed, confidence_growth, confidence_fall):
        """Creates a new Kalman person tracker

        Parameters
        ----------
        id : string
            person identifier
        x0 : np.array
            initial state vector
        u : np.array
            acceleration vector
        std_acc : np.array
            process noise magnitude
        std_meas : np.array
            standard deviation of measurement for x,y,z
        dt : float
            sampling time
        """
        self.id = id
        self.dt = dt
        self.kf = KalmanFilter(self.dt, x0, u,  std_acc, std_meas)
        self.pos = x0

        self.confidence_growth = confidence_growth
        self.confidence_fall = confidence_fall
        self.confidence = 1

        self.frames_not_detected = 0

        self.previous_updated_pos = x0[0:3]

        self.max_certain_speed = max_certain_speed


    def predict(self):
        """Predicts the next position using the kalman filter

        Returns
        -------
        list
            [x,y,z] of the predicted position
        """
        # work with logaritmic decent maybe?
        # self.confidence -= self.confidence_fall
        self.frames_not_detected += 1
        self.frames_not_detected = min(10,self.frames_not_detected)
        self.confidence = 0.75*log(3.7936678946832 - 0.2793667894683*self.frames_not_detected)
        self.confidence = max(self.confidence, 0)
        print("REDUCE confidence:",self.id,",",self.confidence)
        self.pos = self.kf.predict(self.confidence)
        return self.pos

    def update(self, z):
        """Updates the kalman filter with a new measurement

        Parameters
        ----------
        z : np.array
            position measurement vector

        Returns
        -------
        list
            [x,y,z] of the updated filter position
        """
        if z != []:
            z_vect = np.array([z]).T
            #print("update " + self.id + "with val: " + str(z), "diff:" + str(z_vect - self.previous_updated_pos) + "")
            spd = np.linalg.norm(z_vect - self.previous_updated_pos) / self.dt
            print("SPEED:", self.id, spd,",", self.confidence)
            #if spd * self.confidence < self.max_certain_speed:
            if spd * ( 1 - min(0.9, self.frames_not_detected/10)) < self.max_certain_speed:
                self.frames_not_detected = 0
                self.pos = self.kf.update(z_vect)
                self.confidence += self.confidence_growth
                self.confidence = min(self.confidence, 1)
                print("ADD confidence:",self.id,self.confidence,sep=",")

                self.previous_updated_pos = self.pos
            else:
                print("UPDATE REJECTED: misdetection or sudden jump?")
        return self.pos


