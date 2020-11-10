"""
This module contains a Kalman filter implementation for 3D movement tracking.
The code is based on Rahmad Sadli's Kalman filter implementation for 2D movement, which can be found here:
https://github.com/RahmadSadli/2-D-Kalman-Filter/blob/master/KalmanFilter.py
"""

import numpy as np

class KalmanFilter(object):
    """Kalman filter implementation for 3D movement tracking

    Attributes
    ----------
        x : np.array
            state vector: x, y, z, dx/dt, dy/dt, dz/dt

    """
    def __init__(self, dt, x0, u, std_acc, std_meas):
        """Creates a new Kalman filter

        Parameters
        ----------
        dt : float
            sampling time
        x0 : np.array
            initial state vector
        u : np.array
            acceleration vector
        std_acc : np.array
            acceleration standard deviation
        std_meas : np.array
            standard deviation of measurement for x,y,z
        """

        # Define sampling time
        self.dt = dt

        self.std_acc = std_acc

        # Define the  control input variables
        self.u = u

        # Intial State
        self.x = x0

        # Define the State Transition Matrix A
        self.A = np.block(
            [[np.eye(3), self.dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]]
        )

        # Define the Control Input Matrix B
        self.B = np.block(
            [[np.eye(3) * (1 / 2) * (self.dt) ** 2], [np.eye(3) * self.dt]]
        )

        # Define Measurement Mapping Matrix
        self.H = np.block([[np.eye(3), np.zeros((3, 3))]])

        #Initial Process Noise Covariance
        self.Q = self.std_acc ** 2 * np.block(
            [
                [
                    ((self.dt ** 4) / 4) * np.eye(3),
                    ((self.dt ** 3) / 2) * np.eye(3),
                ],
                [((self.dt ** 3) / 2) * np.eye(3), self.dt ** 2 * np.eye(3)],
            ]
        )

        #Initial Measurement Noise Covariance
        self.R = np.matrix([[std_meas[0]**2,0,0],
                           [0, std_meas[1]**2,0],
                           [0,0,std_meas[2]**2]])

        #Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])


    def update_dt(self, dt):
        """ Updates the Kalman matrices A, B, Q with a new sampling rate

        Parameters
        ----------
        dt : float
            new sampling rate
        """

        # Define sampling time
        self.dt = dt

       # Define the State Transition Matrix A
        self.A = np.block(
            [[np.eye(3), self.dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]]
        )

        # Define the Control Input Matrix B
        self.B = np.block(
            [[np.eye(3) * (1 / 2) * (self.dt) ** 2], [np.eye(3) * self.dt]]
        )

        #Initial Process Noise Covariance
        self.Q = self.std_acc ** 2 * np.block(
            [
                [
                    ((self.dt ** 4) / 4) * np.eye(3),
                    ((self.dt ** 3) / 2) * np.eye(3),
                ],
                [((self.dt ** 3) / 2) * np.eye(3), self.dt ** 2 * np.eye(3)],
            ]
        )

    def predict(self):
        """Predicts next X,Y,Z coordinates 

        Returns
        -------
        list
            [x,y,z] predicted position
        """
        # Refer to :Eq.(9) and Eq.(10)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795

        # Update time state
        #x_k =Ax_(k-1) + Bu_(k-1)     Eq.(9)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # Calculate error covariance
        # P= A*P*A' + Q               Eq.(10)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:3]


    def update(self, z):
        """ 
        Updates the filter with a new X,Y,Z measurement

        Returns
        -------
        list
            [x,y,z] updated filter position
        """

        # Refer to :Eq.(11), Eq.(12) and Eq.(13)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795
        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  #Eq.(11)

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))   #Eq.(12)

        I = np.eye(self.H.shape[1])

        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P   #Eq.(13)
        return self.x[0:3]

