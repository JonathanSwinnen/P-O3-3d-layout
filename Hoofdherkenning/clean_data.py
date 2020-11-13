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
                boxes += [box["points"]["exterior"][0] + box["points"]["exterior"][1]]
        data[full_path[:-5].replace("ann","img")] = boxes

# Saving the objects:
with open('clean_anotations.pkl', 'wb') as f:
    pickle.dump(data, f)

## Getting back the objects:
#with open('objs.pkl', "rb") as f:
#    obj0, obj1, obj2 = pickle.load(f)