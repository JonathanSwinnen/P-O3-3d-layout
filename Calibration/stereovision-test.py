from stereovision.blockmatchers import StereoSGBM, StereoBM
from stereovision.calibration import StereoCalibrator, StereoCalibration
import cv2 as cv
import glob
from matplotlib import pyplot as plt

from stereovision.stereo_cameras import CalibratedPair

calibrator = StereoCalibrator(6, 9, 5.3, (640,480))

images_right = glob.glob('pic/RIGHT/*.png')
images_left = glob.glob('pic/LEFT/*.png')
images_left.sort()
images_right.sort()

for i, fname in enumerate(images_right):
    img_l = cv.imread(images_left[i])
    img_r = cv.imread(images_right[i])

    calibrator.add_corners((img_l, img_r))

calibration = calibrator.calibrate_cameras()
calib = StereoCalibration(calibration)
calibration.export("out")

print("\n m")
print(calib.cam_mats)
print("\n d")
print(calib.dist_coefs)
print("\n r")
print(calib.rot_mat)
print("\n t")
print(calib.trans_vec)
print("\n e")
print(calib.e_mat)
print("\n f")
print(calib.f_mat)
print("\n r1 r2")
print(calib.rect_trans)
print("\n p1 p2")
print(calib.proj_mats)
print("\n q")
print(calib.disp_to_depth_mat)




image_pair = [cv.imread("pic/LEFT/opencv_frame_3.png"), cv.imread("pic/RIGHT/opencv_frame_2.png")]

plt.imshow(image_pair[0],'gray')
plt.show()

block_matcher = StereoSGBM()


camera_pair = CalibratedPair(None,
                             calib,
                             block_matcher)
rectified_pair = camera_pair.calibration.rectify(image_pair)
points = camera_pair.get_point_cloud(rectified_pair)
points = points.filter_infinity()
points.write_ply("points.ply")



