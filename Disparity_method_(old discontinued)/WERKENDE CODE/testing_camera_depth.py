import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
window_size = 3
min_disp = 0
num_disp = 112 - min_disp
stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                              numDisparities=num_disp,
                              blockSize=2,
                              P1=8 * 3 * window_size ** 2,
                              P2=32 * 3 * window_size ** 2,
                              disp12MaxDiff=0,
                              uniquenessRatio=15,
                              speckleWindowSize=100,
                              speckleRange=32
                              )
# rectify images
REMAP_INTERPOLATION = cv.INTER_LINEAR
DEPTH_VISUALIZATION_SCALE = 2048

calibration = np.load("out/calib_out.npz", allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])

cam = cv.VideoCapture(1)
cam_2 = cv.VideoCapture(2)

img_counter = 0

while True:
    ret, imgR = cam.read()
    ret_2, imgL = cam_2.read()
    if not ret:
        print("failed to grab frame")
        #break
#       cv2.imshow("test", frame)
    elif not ret_2:
        print("failed to grab frame_2")
        #break
    else:
        fixedLeft = (cv.remap(imgL, leftMapX, leftMapY, REMAP_INTERPOLATION))
        fixedRight = (cv.remap(imgR, rightMapX, rightMapY, REMAP_INTERPOLATION))

        grayLeft = cv.cvtColor(fixedLeft, cv.COLOR_BGR2GRAY)
        grayRight = cv.cvtColor(fixedRight, cv.COLOR_BGR2GRAY)

        disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
        rstereo = cv.ximgproc.createRightMatcher(stereo)
        dispr = rstereo.compute(imgR, imgL).astype(np.float32) / 16.0
        wls_filter = cv.ximgproc.createDisparityWLSFilter(stereo);
        wls_filter.setLambda(8000);
        wls_filter.setSigmaColor(2);
        dispf = wls_filter.filter(disp, imgL, disparity_map_right=dispr);
        faces = face_cascade.detectMultiScale(grayLeft, 1.2, 6)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv.rectangle(dispf, (x, y), (x + w, y + h), (255, 0, 0), 2)
            stuffInRect = dispf[y:y+h, x:x+w] / 255.0 * num_disp
            m = 0
            k = 0
            lst = []
            for i in stuffInRect:
                for j in i:
                    if(np.isfinite(j) and j < num_disp):
                        lst.append(j)
            m = np.mean(lst)
            print("m", m)

            ih, iw = fixedLeft.shape[:2]
            f = 0.6*iw
            s=1
            zc = f*0.25/m * s
            xc = round(zc*x/f,2)
            yc = round(zc*y/f,2)
            zc = round(zc, 2)
            cv.putText(fixedLeft,str((xc, yc, zc)),(x,y), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, cv.LINE_AA)

        cv.imshow("img left", cv.pyrDown(fixedLeft))
        cv.imshow("img right", cv.pyrDown(fixedRight))
        cv.imshow("disp left", cv.pyrDown((disp-min_disp)/num_disp))
        cv.imshow("disp right", cv.pyrDown((-dispr-min_disp)/num_disp))
        cv.imshow("disp filtered", cv.pyrDown(dispf-min_disp)/num_disp)
    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
cam.release()
cam_2.release()
cv.destroyAllWindows()