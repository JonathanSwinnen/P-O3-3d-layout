import pickle
import os
import json
from itertools import chain


data = dict()
paths = ('./data/apart_0/', './data/meer_pers_0/', './data/zittend_0/', './data/apart_1/', './data/meer_pers_1/', './data/zittend_1/')

for root, dirs, files in chain.from_iterable(os.walk(os.path.join(path, "ann/")) for path in paths):
    for file in files:
        full_path = os.path.join(root, file)
        with open(full_path) as f:
            all_boxes = json.load(f)["objects"]
            boxes = []
            for box in all_boxes:
                x1, y1 = box["points"]["exterior"][0]
                x2, y2 = box["points"]["exterior"][1]
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                # xmin, ymin, xmax, ymax
                boxes.append([x1, y1, x2, y2])
        data[full_path[:-5].replace("ann","img")] = boxes

# Saving the objects:
with open('clean_annotations.pkl', 'wb') as f:
    pickle.dump(data, f)

## Getting back the objects:
#with open('objs.pkl', "rb") as f:
#    obj0, obj1, obj2 = pickle.load(f)