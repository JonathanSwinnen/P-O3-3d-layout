import numpy as np
import cv2
import os
import imutils

absolute_path_vid = os.path.join(os.getcwd(), 'p_and_o_3', 'Method_Intersection', 'data', 'videos', 'output_apart_0.avi')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap0 = cv2.VideoCapture(absolute_path_vid)



while True:
    ret, img = cap0.read()	
    
    if (type(img) == type(None)):
        break
    

    image = imutils.resize(img, 
                       width=min(800, img.shape[1])) 

    locations, weights = hog.detectMultiScale(image, winStride=(8,8), padding=(7,7), scale=1.2)

    for(a,b,c,d) in locations:
        cv2.rectangle(image,(a,b),(a+c,b+d),(0,255,210),4)
    
    cv2.imshow('video', image)
    
    if cv2.waitKey(33) == 27:
        break


