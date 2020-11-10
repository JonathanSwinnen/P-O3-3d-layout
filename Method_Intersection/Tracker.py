from filterpy.kalman import KalmanFilter
import numpy as np
import time


class Tracker:
    def __init__(self, acceleration_covariance):
        # TODO: initialize values
        self.person_dict = dict()
        self.state_transition = np.zeros((6, 6))
        self.control_tansition = np.zeros((6, 3))
        self.noise_covariance = np.zeros((6, 6))
        self.measurement_matrix = np.block([[np.eye(3), np.zeros((3, 3))]])
        self.acceleration_covariance = acceleration_covariance

    def add_person(self, id, position):

        self.person_dict[id] = KalmanFilter(dim_x=6, dim_z=3, dim_u=3)
        self.person_dict[id].x = np.array(
            [[position[0]], position[1], position[2], [0], [0], [0]]
        )
        print(self.person_dict[id].x)

    def remove_person(self, id):
        del self.person_dicht[id]

    def update_matrices(self, elapsed_time):
        self.state_transition = np.block(
            [[np.eye(3), elapsed_time * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]]
        )
        print(self.state_transition)

        self.control_tansition = np.block(
            [[np.eye(3) * (1 / 2) * (elapsed_time) ** 2], [np.eye(3) * elapsed_time]]
        )
        print(self.control_tansition)
        self.noise_covariance = self.acceleration_covariance ** 2 * np.block(
            [
                [
                    ((elapsed_time ** 4) / 4) * np.eye(3),
                    ((elapsed_time ** 3) / 2) * np.eye(3),
                ],
                [((elapsed_time ** 3) / 2) * np.eye(3), elapsed_time ** 2 * np.eye(3)],
            ]
        )
        print(self.noise_covariance)

    def add_labels(self, XYZ, elapsed_time):
        self.update_matrices(elapsed_time)
        for id in self.person_dict:
            self.person_dict[id].predict(
                2, self.control_tansition, self.state_transition, self.noise_covariance
            )
            self.person_dict[id].update(XYZ)
            print(self.person_dict[id].x_post)


tracker = Tracker(1)

position = np.array([[1], [1], [1]])
tracker.add_person(1,position)

elapsed_time = 0.1

for i in range(10):
    tracker.add_labels(position, elapsed_time)
    position += np.array([[1], [1], [0]])
