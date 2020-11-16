import cv2


class Detector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_frame(self, image):
        """recognizes a central point of a person in an image

        Args:
            image: the image where the person is in
            one_or_two: camera one or camera two

        returns:
            the point on the image
        """

        locations, _ = self.hog.detectMultiScale(
            image, winStride=(8, 8), padding=(16, 16), scale=1.07
        )

        #   points (x,y)
        coordinates = []
        # if not len(faces) == 0:
        for (x, y, w, h) in locations:
            #  center point of bounding box:
            # coordinates += [(x + w // 2, y + h // 2)]
            #   center of x, 1/4 of y:
            coordinates += [(x + w // 2, y + h // 4)]

        return coordinates

    def detect_both_frames(self, left_frame, right_frame):
        coordinates_left = self.detect_frame(left_frame)
        coordinates_right = self.detect_frame(right_frame)

        # TODO: better filter
        if not coordinates_left:
            coordinates_right = []
        if not coordinates_right:
            coordinates_left = []

        return coordinates_left, coordinates_right