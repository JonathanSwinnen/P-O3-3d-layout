import cv2 as cv
import os
from itertools import chain
from math import floor
import time
import numpy as np


def combine_images(paths):
    for path in paths:
        to_path = './combined_data/' + path + '/'
        from_path_0 = './raw_data/' + path + '_0/img/'
        from_path_1 = './raw_data/' + path + '_1/img/'
        try:
            os.mkdir(to_path)
        except FileExistsError:
            # directory already exists
            pass

        path, dirs, files = next(os.walk(from_path_0))
        file_count = len(files)
        print(to_path)
        for i in range(file_count):
            img1 = cv.imread(from_path_1 + "im_" + str(i) + ".png")
            img2 = cv.imread(from_path_0 + "im_" + str(i) + ".png")
            cv.imwrite(to_path + "im_" + str(i) + ".png", np.concatenate((img1, img2), axis=0))



def compress_images(paths, SCALE_DOWN_FACTOR = 1):
    """
    Compressing the raw images with dimension of 1980x1080 to a more processable size.
    The adjusted images are stored in: './adjusted_data/'a_map'/img_xx.png'

    :param paths: all directories where images are processed.
    :param SCALE_DOWN_FACTOR: factor of how much the images are compressed.
    """

    count = 0
    start = time.time()
    print("Starting compression of all images in " + str(len(paths)) +
          " directories with scale down factor " + str(SCALE_DOWN_FACTOR) + ".")

    # loop over every image file in given directories
    for root, dirs, files in chain.from_iterable(os.walk(os.path.join(path, "img/")) for path in paths):
        # print progress
        print(root)
        for file in files:
            count += 1
            new_path = os.path.join(root, file).replace("/data/", "/adjusted_data/").replace("/img", "")
            cv.imwrite(new_path,  # store in new_path (./adjusted_data/'a_map'/img_xx.png)
                       cv.resize(
                           cv.imread(os.path.join(root, file)),  # read images
                           (floor(1920/SCALE_DOWN_FACTOR), floor(1080/SCALE_DOWN_FACTOR))))  # scale down with factor

    end = time.time()
    print("Compressing images finished in " + str("{:.2f}".format(end-start)) + " s. " +
          str(count) + " files processed.\nImages stored at ./adjusted_data/")


if __name__ == "__main__":
    #paths = ('./raw_data/apart_0/', './raw_data/meer_pers_0/', './raw_data/zittend_0/', './raw_data/apart_1/',
             #'./raw_data/meer_pers_1/', './raw_data/zittend_1/', './raw_data/videodata_0/', './raw_data/videodata_1/')
    #compress_images(paths, SCALE_DOWN_FACTOR=4)
    paths = ('apart', 'meer_pers', 'TA', 'TAFELS', 'two', 'zittend')
    combine_images(paths)