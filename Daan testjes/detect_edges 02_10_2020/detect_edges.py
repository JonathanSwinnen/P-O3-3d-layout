import cv2

image = cv2.imread("image_testing.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 35, 125)
cv2.imshow("image", edged)
cv2.waitKey(0)