from numpy.core.fromnumeric import std
import PersonTracker
import numpy as np
from munkres import Munkres

class Tracker():
    """
    Class for a multi-person tracker
    
    Attributes
    ----------
        persons : list
            The list of PersonTrackers that are being tracked
    """

    def __init__(self, u, std_acc, std_meas, dt):
        """Creates a new Tracker

        Parameters
        ----------
        u : np.array
            acceleration vector
        std_acc : np.array
            acceleration standard deviation
        std_meas : np.array
            measurement standard deviation
        dt : float
            sampling rate
        """
        self.persons = []
        self.u = u
        self.std_acc = std_acc
        self.std_meas = std_meas
        self.dt = dt

        self.positions = dict()


    def add_person(self, id, x0):
        """Adds a person tracker

        Parameters
        ----------
        id : string
            person identifier
        x0 : np.array
            initial state vector
        """
        self.persons.append(PersonTracker.KalmanPersonTracker(id, x0, self.u, self.std_acc, self.std_meas, self.dt))
        

    def rm_person(self, id):
        """removes a person from the tracker

        Parameters
        ----------
        id : string
            person identifier to remove
        """
        self.persons = list(self.persons.filter(lambda p: p.id != id, self.persons))


    def predict(self, dt=None):
        """Predicts new position of all tracked people

        Parameters
        ----------
        dt : float, optional
            time passed since last step, if left None, the previous sampling time will be used, if provided, this will update the sampling time

        Returns
        -------
        dictionary
            dictionary containing person id : [x,y,z] key-value pairs
        """
        self.positions = dict()
        updated_already = False
        # loop through all tracked people
        for person in self.persons:

            # update sampling time if necessary
            if dt is not None and not updated_already:
                person.kf.update_dt(dt)
                updated_already = True
                
            # predict next position
            person.predict()
            self.positions[person.id] = person.pos

        return self.positions


    def update(self, dets):
        """Takes a list of detected points, matches them with the predicted points using the Hungarian method, and uses these matched detections to update the kalman filter

        Parameters
        ----------
        dets : list
            list of all (x,y,z) detected points

        Returns
        -------
        dictionary
            dictionary containing person id : [x,y,z] key-value pairs
        """

        # get indices of matching points with minimal cost combinations using Hungarian method
        if len(dets) == 0:
            return self.positions

        indices, _ = self.get_min_cost(dets)
        # update all kalman filters with matching points
        for (p_num, det_num) in indices:
            # if index is within bounds
            if p_num < len(self.persons) and det_num < len(dets):
                self.persons[p_num].update(dets[det_num])
                self.positions[self.persons[p_num].id] = self.persons[p_num].pos
                
        return self.positions


    def get_min_cost(self, dets):
        """Calculates the minimal cost and indices of detected points matched with predicted points, using the Hungarian method

        Parameters
        ----------
        dets : list
            list of all (x,y,z) detected points

        Returns
        -------
        tuple
            a tuple containing the indices of the best combinations and the cost
        """
        # implementation of Hungarian method
        m = Munkres()

        # i = person index, j = det index
        i, j = 0, 0

        # create cost matrix

        cost_matrix_dim = max(len(self.persons), len(dets))
        cost_matrix = np.zeros((cost_matrix_dim, cost_matrix_dim))
        # loop over all people
        for person in self.persons:
            j = 0
            # loop over all detections
            for det_pos in dets:
                # add cost matrix entry: distance between person prediction point and detection point
                cost_matrix[i][j] = np.linalg.norm( np.array(person.pos) - np.array(det_pos) )
                j += 1
            i += 1

        # compute Hungarian algorithm
        indices = m.compute(cost_matrix)

        # calculate final cost
        cost = 0
        # loop over all best matches
        for (p_num, det_num) in indices:
            if p_num < len(self.persons) and det_num < len(dets):
                # add match cost to total
                cost += cost_matrix[p_num][det_num]
        
        return indices, cost
        








        
        
        


        



