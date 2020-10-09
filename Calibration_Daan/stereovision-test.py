from stereovision.blockmatchers import StereoSGBM, StereoBM
from stereovision.calibration import StereoCalibrator, StereoCalibration
import cv2 as cv
import glob
from matplotlib import pyplot as plt

from stereovision.stereo_cameras import CalibratedPair

calibrator = StereoCalibrator(6, 9, 2.4, (640, 480))

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

print("test")



image_pair = [cv.imread("foto_L.png"), cv.imread("foto_R.png")]

plt.imshow(image_pair[0], 'gray')
# plt.show()

block_matcher = StereoSGBM()


camera_pair = CalibratedPair(None,
                             calib,
                             block_matcher)
print("test")
rectified_pair = camera_pair.calibration.rectify(image_pair)
print("test")
points = camera_pair.get_point_cloud(rectified_pair)
print("test")
points = points.filter_infinity()
print("test")
points.write_ply("points.ply")



