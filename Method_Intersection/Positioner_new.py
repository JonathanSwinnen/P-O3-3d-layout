# Positioner.py rewrite
# NOTE: NIET GETEST!!! Ik heb GEEN idee of die recursieve functie werkt of niet


from munkres import Munkres
import numpy as np
import munkres

class Positioner2:
    def __init__(self, calibration_values, pairing_range):
        """Creates a new Positioner instance

        Parameters
        ----------
        calibration_values : dict
            Dictionary containing calibration values. create this with Calibration.py
        pairing_range : float
            maximum projection error to consider a left and right point combination as a possible pairing
        """
        self.calibration_values = calibration_values
        self.pairing_range = pairing_range


    # TODO: Implement 3D formules
    def get_single_3d_point(self, point_camera_1, point_camera_2):
        pass # momenteel doet dit nog niks


    def get_all_3d_points_with_pairing(self, points_camera_1, points_camera_2, pairings):
        """returns a list of 3D points calculated from points on two cameras, given a pairing configuration

        Parameters
        ----------
        points_camera_1 : list
            a list of points from camera 1
        points_camera_2 : list
            a list of points from camera 2
        pairings : list
            a list filled with tuples (index_1, index_2) containing index pairs that pair points in points_camera_1 to those in points_camera_2,
            meaning they are both projections of the same 3D point in space
        Returns
        -------
        list
            a list with the calculated 3D points
        """
        dets = [] 
        # loop over all pairs
        for pair in pairings:
            # add 3D point calculated from pair of 2D projected points
            dets.append(self.get_single_3d_point(points_camera_1[pair[0]], points_camera_2[pair[1]]))
        return dets


    def get_XYZ(self, points_camera_1, points_camera_2, predictions):
        """Estimates the best guess for XYZ points from detected points on camera 1 and 2, based on a prediction. Detections that yield a big 
        reprojection error are omitted, which means that there might sometimes be less estimated 3D points than expected.

        Parameters
        ----------
        points_camera_1 : list
            a list of points from camera 1
        points_camera_2 : list
            a list of points from camera 2
        predictions : list
            list of prediction points

        Returns
        -------
        list
            a list containing the best estimation of detected 3D points
        """
        
        possible_pairings = []

        # loop over all points and build a list that for every index corresponding to a point in points_camera_1 
        # contains another list of indices of possible pairings from points_camera_2
        for point_camera_1 in points_camera_1:

            # best pairings for this point
            possible_pairings.append([])
            i = 0
            for point_camera_2 in points_camera_2:

                # calculate distance between projected points and projected C1P1 and C2P1 for both images = d1 and d2
                d1, d2 = None, None # TODO: implement

                cost = (d1*d1 + d2*d2) # define the cost as the sum of squared distances to the projected lines

                # points that give a cost below the threshold are possible pairs
                if cost < self.pairing_range:
                    possible_pairings.append(i)

                i += 1

        # get best point combinations and retrieve 3D points
        best_dets = self.get_best_dets_recursively(possible_pairings, points_camera_1, points_camera_2, predictions)
        return best_dets
        
    
    # NOTE: This function is UNTESTED and MIGHT BE UTTER GARBAGE !!!!
    # NOTE: Debugging this will be fun :D /s
    def get_best_dets_recursively(self, pairings_to_choose, points_camera_1, points_camera_2, predictions, current_pairing_index_1=0, chosen_pairings=[]):
        """Recursively finds the best 3D point detections, given a set of pairings to choose from

        Parameters
        ----------
        pairings_to_choose : list
            every element of this list is a nested list which corresponds to an entry in points_camera_1 with the same index. 
            This nested list contains the indices of points from points_camer_2 that can be paired with the entry from points_camera_1
            example: pairings_to_choose = [[2,3],[1]] means that points_camera_1[0] can be paired with either points_camera_2[2] or [3], 
            and that points_camera_1[2] can only be paired with points_camera_2[1]
        points_camera_1 : list
            a list of points from camera 1
        points_camera_2 : list
            a list of points from camera 2
        predictions : list
            list of prediction points
        current_pairing_index_1 : int, optional
            current index to look for pairing options, used for recursion, by default 0
        chosen_pairings : list, optional
            currently chosen pairings, used for recursion, by default []

        Returns
        -------
        list
            best 3D point detections from given pairing options
        """

        # skip leading empty entries
        while current_pairing_index_1 < len(pairings_to_choose) and len(pairings_to_choose[current_pairing_index_1]) == 0:
            current_pairing_index_1 += 1
        
        # no more pairings to choose => calculate and return cost & dets
        if len(pairings_to_choose[current_pairing_index_1:]) == 0:
            # get 3D points from chosen pairings
            dets = self.get_all_3d_points_with_pairing(points_camera_1, points_camera_2, chosen_pairings)
            # get cost from 3D points
            cost = self.get_mean_dets_vs_prediction_cost(dets, predictions)
            return cost, dets
        
        # loop over all pairing possibilities i for current index of points_camera_1 to get minimum cost
        min_cost = None
        best_dets = None
        for i in pairings_to_choose[current_pairing_index_1]:
            # new chosen pairings
            new_chosen_pairings = list(chosen_pairings) # copy
            new_chosen_pairings.append(current_pairing_index_1, i) # add this pair
            # new next pairings to choose -> move to next entry
            new_pairings_to_choose = list(pairings_to_choose[current_pairing_index_1+1:])
            # remove duplicates of i from other pairing possibilities, so no camera point can be used twice, !!! CAN LEAD TO SKIPPED POINTS
            for next_pairing in new_pairings_to_choose:
                if i in next_pairing: 
                    next_pairing.remove(i)
            # recursion -> get dets from best pairing sequence after this one
            cost, dets = self.test_pairings_recursively(new_pairings_to_choose, points_camera_1, points_camera_2, predictions, current_pairing_index_1+1, new_chosen_pairings)
            # if this pairing & best next pairing together are optimal, set new best dets & cost
            if min_cost is None or cost < min_cost:
                min_cost = cost
                best_dets = dets

        # return dets from best pairing sequence
        return best_dets


    def get_mean_dets_vs_prediction_cost(self, dets, predictions):
        """returns the minimal mean cost of the distance between a given list of detected 3D points and a prediction
        through the hungarian method.

        Parameters
        ----------
        dets : list
            a list of detected 3D points
        predictions : list
            a list of predictions for the 3D points

        Returns
        -------
        float
            the minimal cost
        """
        # implementation of Hungarian method
        m = Munkres()

        # i = person index, j = det index
        i, j = 0, 0
        # create cost matrix
        cost_matrix_dim = max(len(predictions), len(dets))
        cost_matrix = np.zeros((cost_matrix_dim, cost_matrix_dim))
        # loop over all predictions
        for prediction_pos in predictions:
            j = 0
            # loop over all detections
            for det_pos in dets:
                # add cost matrix entry: distance between prediction point and detection point
                cost_matrix[i][j] = np.linalg.norm( np.array(prediction_pos) - np.array(det_pos) )
                j += 1
            i += 1

        # compute Hungarian algorithm
        indices = m.compute(cost_matrix)

        # calculate final cost
        cost = 0
        # loop over all best matches
        for (p_num, det_num) in indices:
            if p_num < len(predictions) and det_num < len(dets):
                # add match cost to total
                cost += cost_matrix[p_num][det_num]
        cost /= len(indices)
        return cost




            
            

            









    def get_XYZ(
        self,
        points_camera_1,
        points_camera_2,
    ):
        """Determines the XYZ point of the given point seen by two camera's

        Notes:
            - function receives, works with, and returns values in [meter]
            - camera one is positioned above the axis origin
            - direction camera1,camera2 is the direction of the x-axis
            - y axis is pointed lateral to x, in the direction the cameras are looking at
            - z axis is pointed up

        Args:
            self.calibration_values["image_size"]: Size of the images (#pixels x, #pixels y)
            fov: field of view of the camera (horizontally)
            point_camera_1 /2 : the pixel on the images where the point is seen
            self.calibration_values["coord_1"] and self.calibration_values["coord_2"]: positions of cameras
            self.calibration_values["dir_1"] and self.calibration_values["dir_2"]: direction of cameras

        Returns:
            the calculated XYZ points, in a list of np.arrays
        """

        XYZ_POINTS = []


        # TODO: optimise: delete already calculated points

        for afb_pos_1 in points_camera_1:
            #   for every point recognized in camera1's image do:
            #   get the field of view in Radians

            #   determine line between self.calibration_values["coord_1"] and point on projected image plane 1 (point that has to be calculated)
            #   projected image plane 1
            #       take a plane within the field of view at a distance d from camera_1
            d = 0.5

            #   center point of plane:
            M1 = (
                d * self.calibration_values["dir_1"]
                + self.calibration_values["coord_1"]
            )

            #   now we can determine the location of the recognized point in space (P1)
            #       first, determine the middle of the image:
            M_afbeelding = np.array(
                [
                    self.calibration_values["image_size"][0] / 2,
                    self.calibration_values["image_size"][1] / 2,
                ]
            )

            # P1 can be calculated!
            P1 = (
                M1
                + (afb_pos_1[0] - M_afbeelding[0]) * self.calibration_values["x1"]
                + (afb_pos_1[1] - M_afbeelding[1]) * self.calibration_values["y1"]
            )

            #   Now we need to find line C1P1 in the second image
            #       thus, we need a mathematical representation of the second 'imagescreen' plane bv2
            #               we know where the camera faces (self.calibration_values["dir_2"]) and distance d
            M2 = self.calibration_values["coord_2"] + (
                d * self.calibration_values["dir_2"]
            )

            #   by knowing two points of line C1P1 on the second image, we can determine the line completely:
            #                   intersection of C1C2 and the plane bv2
            #                   and of P1C2 and the plane bv2
            #   for this step, we need two linear independant vectors of bv2
            #           we'll need these later, so let's calculate x2,y2
            #           x2 is allong the intersection of horizontal plane and image plane 2

            #   To find the intersections, two linear systems need to be computed (see calculations on paper (later in pdf))
            #   this linear system determines the intersection of line C1C2 and image plane 2
            #
            #       equation: k*C1C2 + self.calibration_values["coord_1"] = l*x2 + m*y2 + M2
            #           where: k, l and m variable parameters, self.calibration_values["coord_1"] and M2 known points and C1C2, x2 and y2 directions
            C1C2 = (
                self.calibration_values["coord_2"] - self.calibration_values["coord_1"]
            )
            A = np.array(
                [
                    [
                        -C1C2[0],
                        self.calibration_values["x2"][0],
                        self.calibration_values["y2"][0],
                    ],
                    [
                        -C1C2[1],
                        self.calibration_values["x2"][1],
                        self.calibration_values["y2"][1],
                    ],
                    [
                        -C1C2[2],
                        self.calibration_values["x2"][2],
                        self.calibration_values["y2"][2],
                    ],
                ]
            )
            b = self.calibration_values["coord_1"] - M2

            # solve b = Ax where x[0] is k:
            solution_1 = np.linalg.solve(A, b)

            #   the first solution is the coefficient k needed in k*C1C2 + self.calibration_values["coord_1"] to reach the plane
            #       thus, by knowing k, we know the first of two intersection points: IP1
            IP1 = solution_1[0] * C1C2 + self.calibration_values["coord_1"]

            #   Now the same solution method for P1C2's intersection with the second image plane
            #   equation: k*P1C2 + P1 = l*x2 + m*y2 + M2

            P1C2 = self.calibration_values["coord_2"] - P1
            A = np.array(
                [
                    [
                        -P1C2[0],
                        self.calibration_values["x2"][0],
                        self.calibration_values["y2"][0],
                    ],
                    [
                        -P1C2[1],
                        self.calibration_values["x2"][1],
                        self.calibration_values["y2"][1],
                    ],
                    [
                        -P1C2[2],
                        self.calibration_values["x2"][2],
                        self.calibration_values["y2"][2],
                    ],
                ]
            )
            b = P1 - M2

            solution_2 = np.linalg.solve(A, b)
            IP2 = solution_2[0] * P1C2 + P1

            #   Two points are known, thus the Line of sight of camera one, projected on the
            #       second image plane, is known too:

            Line_of_sight_1 = IP2 - IP1
            Line_of_sight_1_norm = np.linalg.norm(
                Line_of_sight_1
            )  #    (normalizing the vector)
            Line_of_sight_1 = Line_of_sight_1 / Line_of_sight_1_norm

            #   Now we know where the line-of-sight of camera1 towards said person is,
            #       we need to find the closest recognized point on camera two's image
            #       the intersection of the line between this point and self.calibration_values["coord_2"] and the first line C1P1 is the XYZ point!

            #   P2 is the point of the person in camera2's image
            #   for every detected point on image_2: find the distance, and determine which is most likely to be the right one
            distance = None
            closest_position = None
            shortest_distance = None

            for recognized_pos_2 in points_camera_2:
                P2 = (
                    M2
                    + (recognized_pos_2[0] - M_afbeelding[0])
                    * self.calibration_values["x2"]
                    + (recognized_pos_2[1] - M_afbeelding[1])
                    * self.calibration_values["y2"]
                )
                #   find distance of P2 to Line_of_Sight_1, via formula:
                distance = np.linalg.norm(
                    np.cross(P2 - IP1, P2 - IP2)
                ) / np.linalg.norm(Line_of_sight_1)
                if not shortest_distance or distance <= shortest_distance : #- self.uncertainty_range
                    shortest_distance = distance
                    closest_position = recognized_pos_2
                # elif (
                #     shortest_distance - self.uncertainty_range
                #     < distance
                #     < shortest_distance + self.uncertainty_range
                # ):
                #     # TODO: zoeken welk de beste is via cost
                #     pass

            # closest match is found: closest_position
            P2 = (
                M2
                + (closest_position[0] - M_afbeelding[0])
                * self.calibration_values["x2"]
                + (closest_position[1] - M_afbeelding[1])
                * self.calibration_values["y2"]
            )

            #   shortest line between this point and line of sight:
            #   direction:
            connecting_line = np.cross(
                Line_of_sight_1, self.calibration_values["dir_2"]
            )

            #   linear system to determine the intersection of the connecting line with line of sight:
            #       equation: k*Line_of_sight + IP2 = m*connecting_line + P2
            A = np.array(
                [
                    [-Line_of_sight_1[0], connecting_line[0]],
                    [-Line_of_sight_1[1], connecting_line[1]],
                    [-Line_of_sight_1[2], connecting_line[2]],
                ]
            )
            b = IP2 - P2

            solution_3 = np.linalg.lstsq(A, b, rcond=-1)
            Matching_point_bv2 = solution_3[0][0] * Line_of_sight_1 + IP2

            #   Now we have everything mathematically needed to calculate the intersection of the two Line_of_sight_1's
            Line_1 = P1 - self.calibration_values["coord_1"]
            Line_2 = Matching_point_bv2 - self.calibration_values["coord_2"]

            A = np.array(
                [
                    [-Line_1[0], Line_2[0]],
                    [-Line_1[1], Line_2[1]],
                    [-Line_1[2], Line_2[2]],
                ]
            )
            b = self.calibration_values["coord_1"] - self.calibration_values["coord_2"]

            solution_4 = np.linalg.lstsq(A, b, rcond=-1)
            XYZ_POINTS += [
                (solution_4[0][0] * Line_1 + self.calibration_values["coord_1"])
            ]

        return XYZ_POINTS

    def reprojectPoint(self, xyz):
        # TODO: vervang deze placeholder code met projectie

        xyz = np.array([xyz[0][0], xyz[1][0], xyz[2][0]])
        # xyz = np.array([xyz[0][0], xyz[1][0], 1.50])

        # Center of each image plane:
        d = 0.5
        M1 = self.calibration_values["coord_1"] + d * self.calibration_values["dir_1"]

        M2 = self.calibration_values["coord_2"] + d * self.calibration_values["dir_2"]


        Line_sight_1 = xyz - self.calibration_values["coord_1"] 
        Line_sight_2 = xyz - self.calibration_values["coord_2"]
        # Calculate intersection of the line from xyz to each image sensor with the image planes:
        # print([
                # [
                    # Line_sight_1[0],
                    # self.calibration_values["x1"][0],
                    # self.calibration_values["y1"][0],
                # ],
                # [
                    # Line_sight_1[1],
                    # self.calibration_values["x1"][1],
                    # self.calibration_values["y1"][1],
                # ],
                # [
                    # Line_sight_1[2],
                    # self.calibration_values["x1"][2],
                    # self.calibration_values["y1"][2],
                # ]
            # ])
        A1 = np.array(
            [
                [
                    Line_sight_1[0],
                    self.calibration_values["x1"][0],
                    self.calibration_values["y1"][0],
                ],
                [
                    Line_sight_1[1],
                    self.calibration_values["x1"][1],
                    self.calibration_values["y1"][1],
                ],
                [
                    Line_sight_1[2],
                    self.calibration_values["x1"][2],
                    self.calibration_values["y1"][2],
                ]
            ]
        ) 

        A2 = np.array(
            [
                [
                    Line_sight_2[0],
                    self.calibration_values["x2"][0],
                    self.calibration_values["y2"][0],
                ],
                [
                    Line_sight_2[1],
                    self.calibration_values["x2"][1],
                    self.calibration_values["y2"][1],
                ],
                [
                    Line_sight_2[2],
                    self.calibration_values["x2"][2],
                    self.calibration_values["y2"][2],
                ]
            ]
        )

        b1 = xyz-M1
        b2 = xyz-M2

        solution_1 = np.linalg.solve(A1, b1)

        solution_2 = np.linalg.solve(A2, b2)




        middle_pixel = (self.calibration_values["image_size"][0]//2, self.calibration_values["image_size"][1]//2)
        XY1 = (middle_pixel[0] + solution_1[1], middle_pixel[1] + solution_1[2])
        XY2 = (middle_pixel[0] + solution_2[1], middle_pixel[1] + solution_2[2])

        return (XY1, XY2)
