import cv2


class Detector:
    def __init__(self):
        pass

    def detect_frame(self, image):
        """recognizes a central point of a person in an image

        Args:
            image: the image where the person is in
            one_or_two: camera one or camera two

        returns:
            the point on the image
        """

        #   Select the wanted haarcascade path right here:
        cascPath = r"C:\Users\peete\Documents\Burgi\2020-2021\Daan_bestanden\detection_daan\haarcascades\haarcascade_frontalface_alt2.xml"

        # Create the haar cascade
        faceCascade = cv2.CascadeClassifier(cascPath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect Faces
        faces = faceCascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=2, minSize=(10, 10)
        )

        # flags = cv2.CV_HAAR_SCALE_IMAGE)

        #   points (x,y)
        coordinates = []
        # if not len(faces) == 0:
        for (x, y, w, h) in faces:
            coordinates += [(x + w // 2, y + h // 2)]

        return coordinates

    def detect_both_frames(self, left_frame, right_frame):
        coordinates_left = self.detect_frame()
        coordinates_right = self.detect_frame()

        # TODO: better filter
        if not coordinates_left:
            coordinates_right = []
        if not coordinates_right:
            coordinates_left = []

        return coordinates_left, coordinates_right