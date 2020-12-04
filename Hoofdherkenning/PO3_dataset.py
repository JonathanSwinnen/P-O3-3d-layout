import os
import torch
from PIL import Image

from itertools import chain
import pickle
import pytorch_files.transforms as T


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class PO3Dataset(object):
    """
    Custom dataset class for training a head detector model.
    """
    def __init__(self, paths, transforms, has_sub_maps=False, ann_path="./adjusted_data/clean_ann_scaled.pkl"):
        # paths where images are located
        self.paths = paths

        # are images located in submap: "path/img/img_xx.png" (True)
        # or not: "path/img_xx.png" (False)
        self.sub_maps = has_sub_maps

        # transformer
        self.transforms = transforms

        # load annotations from given file_path
        f = open(ann_path, 'rb')
        self.ann = pickle.load(f)
        f.close()

        # load all image files from given paths, sorting them to
        # ensure that they are aligned
        self.imgs = []

        submap = ""
        if self.sub_maps:
            submap = "img/"
            
        for root, dirs, files in chain.from_iterable(os.walk(os.path.join(path, submap)) for path in paths):
            for file in files:
                # add image
                self.imgs += [os.path.join(root, file)]
        self.imgs = sorted(self.imgs)

        print("Total images in dataset:", len(self.imgs))

    def __getitem__(self, idx):
        # load an image
        image_id = torch.tensor([idx])
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")

        # get bounding box coordinates for each mask
        boxes = self.ann[img_path][0]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # define classes
        labels = torch.tensor(self.ann[img_path][1], dtype=torch.int64)  # background

        # calc area
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