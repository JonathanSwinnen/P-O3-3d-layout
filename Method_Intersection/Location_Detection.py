import numpy as np
import math


def XYZ_Point(
    image_size: np.ndarray,
    fov_horizontal,
    points_camera_1,
    points_camera_2,
    C1,
    C2,
    rg1,
    rg2,
):
    """Determines the XYZ point of the given point seen by two camera's

    Notes:
        - function receives, works with, and returns values in [meter]
        - camera one is positioned above the axis origin
        - direction camera1,camera2 is the direction of the x-axis
        - y axis is pointed lateral to x, in the direction the cameras are looking at
        - z axis is pointed up

    Args:
        image_size: Size of the images (#pixels x, #pixels y)
        fov: field of view of the camera (horizontally)
        point_camera_1 /2 : the pixel on the images where the point is seen
        C1 and C2: positions of cameras
        rg1 and rg2: direction of cameras

    Returns:
        the calculated XYZ points, in a list of np.arrays
    """

    XYZ_POINTS = []

    #   get a normalized vector of both the directions of the cameras
    rg1_norm = np.linalg.norm(rg1)
    rg2_norm = np.linalg.norm(rg2)
    rg1 = rg1 / rg1_norm
    rg2 = rg2 / rg2_norm
    #   get the field of view in Radians
    fov_horizontal_rad = (fov_horizontal / 180) * math.pi
    #   get a diagonal field of view (for ease further)
    convert_to_diag = (math.sqrt(image_size[0] ** 2 + image_size[1] ** 2)) / (
        image_size[0]
    )
    fov = fov_horizontal_rad * convert_to_diag

    for afb_pos_1 in points_camera_1:
        #   for every point recognized in camera1's image do:
        #   get the field of view in Radians

        #   determine line between C1 and point on projected image plane 1 (point that has to be calculated)
        #   projected image plane 1
        #       take a plane within the field of view at a distance d from camera_1
        d = 0.5

        #   center point of plane:
        M1 = d * rg1 + C1

        #   directions of the axis of the image on this plane:
        #   we want to create "normalized" vectors that follow both the axis of the image on this projection plane
        #           "normalized": their length is the projected length of 1 pixel on this plane

        #   x1 is horizontal, lateral to rg1
        horizontal_rg = np.array([0, 0, 1])
        x1 = np.cross(horizontal_rg, rg1)

        #   we say x1's direction to have a positive x-value
        if x1[0] < 0:
            x1 = -x1

        #   calculation of the size of a pixel:
        size_pixel = (2 * d * np.tan(fov / 2)) / (
            math.sqrt((image_size[0]) ** 2 + (image_size[1]) ** 2)
        )

        #   normalize x1 to be the same size as a pixel
        x1_norm = np.linalg.norm(x1)
        x1 = x1 / x1_norm  #   x is 1m long
        x1 = x1 * size_pixel  #    x is 1 pixel long

        #   y1 is lateral to rg1 and in a vertical plane
        #       this vertical plane has rg1 and the vertical vector in it:
        vertical_rg = np.cross(rg1, np.array([0, 0, 1]))
        y1 = np.cross(vertical_rg, rg1)

        #   we say y1's direction to have a negative y-value
        if y1[1] > 0:
            y1 = -y1

        #   normalize y1 as the size of one pixel
        y1_norm = np.linalg.norm(y1)
        y1 = y1 / y1_norm  #   is nu 1m lang
        y1 = y1 * size_pixel  #    is nu 1 pixel lang

        #   now we can determine the location of the recognized point in space (P1)
        #       first, determine the middle of the image:
        M_afbeelding = np.array([image_size[0] / 2, image_size[1] / 2])

        # P1 can be calculated!
        P1 = (
            M1
            + (afb_pos_1[0] - M_afbeelding[0]) * x1
            + (afb_pos_1[1] - M_afbeelding[1]) * y1
        )

        #   Now we need to find line C1P1 in the second image
        #       thus, we need a mathematical representation of the second 'imagescreen' plane bv2
        #               we know where the camera faces (rg2) and distance d
        M2 = C2 + (d * rg2)

        #   by knowing two points of line C1P1 on the second image, we can determine the line completely:
        #                   intersection of C1C2 and the plane bv2
        #                   and of P1C2 and the plane bv2
        #   for this step, we need two linear independant vectors of bv2
        #           we'll need these later, so let's calculate x2,y2
        #           x2 is allong the intersection of horizontal plane and image plane 2

        x2 = np.cross(horizontal_rg, rg2)

        #   x2's x-coordinate is positive:
        if x2[0] < 0:
            x2 = -x2
        x2_norm = np.linalg.norm(x2)
        x2 = x2 / x2_norm  #   now it is 1m long

        #   now let's find y2
        #       the plane made by C2M2 and (0,0,1) intersects bv2 allong y2
        C2M2 = M2 - C2
        verticaal_rg_2 = np.cross(C2M2, np.array([0, 0, 1]))
        y2 = np.cross(verticaal_rg_2, rg2)
        if y2[2] > 0:
            y2 = -y2
        y2_norm = np.linalg.norm(y2)
        y2 = y2 / y2_norm  #   now it is 1m long

        #   To find the intersections, two linear systems need to be computed (see calculations on paper (later in pdf))
        #   this linear system determines the intersection of line C1C2 and image plane 2
        #
        #       equation: k*C1C2 + C1 = l*x2 + m*y2 + M2
        #           where: k, l and m variable parameters, C1 and M2 known points and C1C2, x2 and y2 directions
        C1C2 = C2 - C1
        A = np.array(
            [
                [-C1C2[0], x2[0], y2[0]],
                [-C1C2[1], x2[1], y2[1]],
                [-C1C2[2], x2[2], y2[2]],
            ]
        )
        b = C1 - M2

        # solve b = Ax where x[0] is k:
        solution_1 = np.linalg.solve(A, b)

        #   the first solution is the coefficient k needed in k*C1C2 + C1 to reach the plane
        #       thus, by knowing k, we know the first of two intersection points: IP1
        IP1 = solution_1[0] * C1C2 + C1

        #   Now the same solution method for P1C2's intersection with the second image plane
        #   equation: k*P1C2 + P1 = l*x2 + m*y2 + M2

        P1C2 = C2 - P1
        A = np.array(
            [
                [-P1C2[0], x2[0], y2[0]],
                [-P1C2[1], x2[1], y2[1]],
                [-P1C2[2], x2[2], y2[2]],
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
        #       the intersection of the line between this point and C2 and the first line C1P1 is the XYZ point!

        #   P2 is the point of the person in camera2's image
        #   normalize x2 and y2 as the size of one pixel
        x2 = x2 * size_pixel
        y2 = y2 * size_pixel

        #   for every detected point on image_2: find the distance, and determine which is most likely to be the right one
        distance = None
        closest_position = None
        shortest_distance = None

        for recognized_pos_2 in points_camera_2:
            P2 = (
                M2
                + (recognized_pos_2[0] - M_afbeelding[0]) * x2
                + (recognized_pos_2[1] - M_afbeelding[1]) * y2
            )
            #   find distance of P2 to Line_of_Sight_1, via formula:
            distance = np.linalg.norm(np.cross(P2 - IP1, P2 - IP2)) / np.linalg.norm(
                Line_of_sight_1
            )
            if not shortest_distance or distance < shortest_distance:
                shortest_distance = distance
                closest_position = recognized_pos_2

        # closest match is found: closest_position
        P2 = (
            M2
            + (closest_position[0] - M_afbeelding[0]) * x2
            + (closest_position[1] - M_afbeelding[1]) * y2
        )

        #   shortest line between this point and line of sight:
        #   direction:
        connecting_line = np.cross(Line_of_sight_1, rg2)

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
        Line_1 = P1 - C1
        Line_2 = Matching_point_bv2 - C2

        A = np.array(
            [
                [-Line_1[0], Line_2[0]],
                [-Line_1[1], Line_2[1]],
                [-Line_1[2], Line_2[2]],
            ]
        )
        b = C1 - C2

        solution_4 = np.linalg.lstsq(A, b, rcond=-1)
        XYZ_POINTS += [solution_4[0][0] * Line_1 + C1]
    return XYZ_POINTS
