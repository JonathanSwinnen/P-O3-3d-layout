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

        locations, _ = self.hog.detectMultiScale(image, winStride=(8,8), padding=(7,7), scale=1.1)

        #   points (x,y)
        coordinates = []
        # bounding boxes
        boxes = []
        # if not len(faces) == 0:
        for (x, y, w, h) in locations:
            coordinates += [(x + w // 2, y + h // 2)]
            boxes += [(x, y, x+w, y+h)]

        return coordinates, boxes

    def detect_both_frames(self, left_frame, right_frame):
        coordinates_left, boxes_left = self.detect_frame(left_frame)
        coordinates_right, boxes_right = self.detect_frame(right_frame)

        # TODO: better filter
        if not coordinates_left:
            coordinates_right = []
        if not coordinates_right:
            coordinates_left = []

        return coordinates_left, coordinates_right, boxes_left, boxes_right