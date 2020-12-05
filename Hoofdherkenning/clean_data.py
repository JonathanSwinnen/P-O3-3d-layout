import pickle
import os
import json
from itertools import chain
from math import floor
import time

def generate_clean_data_combined(paths):
    data = dict()
    for path in paths:
        from_path_0 = './raw_data/' + path + '_0/ann/'
        from_path_1 = './raw_data/' + path + '_1/ann/'

        pathx, dirs, files = next(os.walk(from_path_0))
        file_count = len(files)
        for i in range(file_count):
            data_boven = None
            data_onder = None
            with open(from_path_1 + 'im_' + str(i) + '.png.json') as f:
                data_boven = json.load(f)["objects"]
            with open(from_path_0 + 'im_' + str(i) + '.png.json') as f:
                data_onder = json.load(f)["objects"]
            boxes, labels = [], []
            for box in data_boven:
                x1, y1 = box["points"]["exterior"][0]
                x2, y2 = box["points"]["exterior"][1]

                # raise error if boxes are incomplete
                if x1 == x2 or y1 == y2:
                    bbox = str(x1) + ", " + str(x2) + ", " + str(y1) + ", " + str(y2)
                    raise Exception('Invalid boundingbox')

                # make sure that the coordinates are in the correct order
                # xmin, ymin, xmax, ymax
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
            # add bbox and adjust with SCALING_DOWN_FACTOR
                boxes.append([x1, y1, x2, y2])

                if box["classId"] == 2707862:
                    labels.append(0)
                elif box["classId"] == 2703929:
                    labels.append(1)
                elif box["classId"] == 2712297:
                    labels.append(2)
                else:
                    raise Exception('Invalid label')

            for box in data_onder:
                x1, y1 = box["points"]["exterior"][0]
                x2, y2 = box["points"]["exterior"][1]

                # raise error if boxes are incomplete
                if x1 == x2 or y1 == y2:
                    bbox = str(x1) + ", " + str(x2) + ", " + str(y1) + ", " + str(y2)
                    raise Exception('Invalid boundingbox')

                # make sure that the coordinates are in the correct order
                # xmin, ymin, xmax, ymax
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                # add bbox and adjust with SCALING_DOWN_FACTOR
                boxes.append([x1, y1+1080, x2, y2+1080])

                if box["classId"] == 2707862:
                    labels.append(0)
                elif box["classId"] == 2703929:
                    labels.append(1)
                elif box["classId"] == 2712297:
                    labels.append(2)
                else:
                    raise Exception('Invalid label')
            if any(j == 0 for j in labels):
                if all(j == 0 for j in labels):
                    labels = [0]
                    boxes = [[0, 0, 1980, 1080*2]]
                else:
                    j = 0
                    run = True
                    while run:
                        if labels[j] == 0:
                            del labels[j]
                            del boxes[j]
                            j -= 1
                        j += 1
                        if len(labels) == j:
                            run = False

            data['./combined_data/' + path + '/im_' + str(i) + '.png'] = (boxes, labels)
    print(data)
    with open("./combined_data/clean_ann_combined.pkl", 'wb') as f:
        pickle.dump(data, f)


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
                    elif box["classId"] == 2712297:
                        # head
                        classes.append(1)
                    else:
                        classes.append(2)

            # store list of all bboxes in dict at the name of the image
            if SUBDIRECOTORY_FOR_IMAGES:
                # images are stored in a subdirectory: './data/a_map/img/img_xx.png'
                img_path = full_path[:-5].replace("/ann", "/img")
            else:
                # no subdirectory: './data/a_map/img_xx.png'
                img_path = full_path[:-5].replace("/ann", "")

            # if adjusted scaling change path
            if SCALING_DOWN_FACTOR != 1:
                img_path = img_path.replace("raw_data", "adjusted_data")

            # store in dict
            data[img_path] = (boxes, classes)

    # Saving the annotations dictionary in the given path
    with open(storing_path, 'wb') as f:
        pickle.dump(data, f)

    end = time.time()
    print("Generating clean data finished in " + str("{:.2f}".format(end-start)) + " s. " +
          str(count) + " files processed.\nData stored at: " + storing_path)


if __name__ == "__main__":
    # paths = ('./raw_data/apart_0/', './raw_data/meer_pers_0/', './raw_data/zittend_0/', './raw_data/apart_1/',
    #          './raw_data/meer_pers_1/', './raw_data/zittend_1/')
    # generate_clean_data(paths)
    paths = ('apart', 'meer_pers', 'TA', 'TAFELS', 'two', 'zittend')
    generate_clean_data_combined(paths)
