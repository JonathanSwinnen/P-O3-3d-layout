import pickle
import os
import json
from itertools import chain
from math import floor
import time


def generate_clean_data(data_paths, storing_path="./adjusted_data/clean_ann_scaled.pkl", SCALING_DOWN_FACTOR=4, SUBDIRECOTORY_FOR_IMAGES=False):
    """
    Generate clean data in the form of a dictionary.
    Original data is stored in  581 .json files --> all in 1 stored dictionary for easy access.

    :param data_paths: list of directories. All files are processed in these directories.
    :param storing_path: path where dictionary will be stored.
    :param SCALING_DOWN_FACTOR: bounding boxes are scaled down by this factor.
    :param SUBDIRECOTORY_FOR_IMAGES
    """

    data = dict()
    count = 0
    start = time.time()

    # loop over all annotation files
    for root, dirs, files in chain.from_iterable(os.walk(os.path.join(path, "ann/")) for path in data_paths):
        # print progress
        print(root)
        for file in files:
            count += 1
            full_path = os.path.join(root, file)
            # open annotation file ('root/img_xx.png.json')
            with open(full_path) as f:
                # read json
                all_boxes = json.load(f)["objects"]
                boxes = []
                classes = []

                # extract all bounding boxes
                for box in all_boxes:
                    x1, y1 = box["points"]["exterior"][0]
                    x2, y2 = box["points"]["exterior"][1]

                    # raise error if boxes are incomplete
                    if x1 == x2 or y1 == y2:
                        bbox = str(x1) + ", " + str(x2) + ", " + str(y1) + ", " + str(y2)
                        raise Exception('Invalid boundingbox: ' + bbox + " in file '" + full_path + "'.")

                    # make sure that the coordinates are in the correct order
                    # xmin, ymin, xmax, ymax
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1

                    # add bbox and adjust with SCALING_DOWN_FACTOR
                    boxes.append([floor(x1 / SCALING_DOWN_FACTOR),
                                  floor(y1 / SCALING_DOWN_FACTOR),
                                  floor(x2 / SCALING_DOWN_FACTOR),
                                  floor(y2 / SCALING_DOWN_FACTOR)])

                    if box["classId"] == 2707862:
                        # background class
                        classes.append(0)
                    else:
                        # head
                        classes.append(1)

            # store list of all bboxes in dict at the name of the image
            if SUBDIRECOTORY_FOR_IMAGES:
                # images are stored in a subdirectory: './data/a_map/img/img_xx.png'
                img_path = full_path[:-5].replace("/ann", "/img")
            else:
                # no subdirectory: './data/a_map/img_xx.png'
                img_path = full_path[:-5].replace("/ann", "")

            # store in dict
            data[img_path] = (boxes, classes)

    # Saving the annotations dictionary in the given path
    with open(storing_path, 'wb') as f:
        pickle.dump(data, f)

    end = time.time()
    print("Generating clean data finished in " + str("{:.2f}".format(end-start)) + " s. " +
          str(count) + " files processed.\nData stored at: " + storing_path)


if __name__ == "__main__":
    paths = ('./raw_data/apart_0/', './raw_data/meer_pers_0/', './raw_data/zittend_0/', './raw_data/apart_1/',
             './raw_data/meer_pers_1/', './raw_data/zittend_1/', './raw_data/videodata_0/', './raw_data/videodata_1/')
    generate_clean_data(paths)
