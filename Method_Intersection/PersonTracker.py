"""
This file contains a class containing the logic for a person tracker using a Kalman filter.

#TODO: expand algorithm

"""

from KalmanFilter import KalmanFilter
import numpy as np

class KalmanPersonTracker:

    def __init__(self, id, x0, u, std_acc, std_meas, dt):
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

        self.kf = KalmanFilter(dt, x0, u,  std_acc, std_meas)

        self.pos = x0

    def predict(self):
        """Predicts the next position using the kalman filter

        Returns
        -------
        list
            [x,y,z] of the predicted position
        """
        self.pos = self.kf.predict()
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
            
            self.pos = self.kf.update(np.array([z]).T)
        return self.pos
    
    


    

