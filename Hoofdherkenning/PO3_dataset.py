import os
import numpy as np
import torch
from PIL import Image

from itertools import chain
import pickle

class PO3Dataset(object):
    def __init__(self, paths, transforms):
        self.paths = paths
        #print(paths)

        self.transforms = transforms
        f = open('clean_annotations.pkl', 'rb')
        self.ann = pickle.load(f)
        # print(self.ann)
        f.close()
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = []
        total = 0
        for root, dirs, files in chain.from_iterable(os.walk(os.path.join(path, "img/")) for path in paths):
            for file in files:
                file_name = os.path.join(root, file)
                total += 1
                if len(self.ann[file_name]) > 0:
                    self.imgs += [file_name]
        self.imgs = sorted(self.imgs)
        print("total images:", total)
        print("filtered images:", total - len(self.imgs))
        print("usable images:", len(self.imgs))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")

        # get bounding box coordinates for each mask
        boxes = self.ann[img_path]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])


        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)