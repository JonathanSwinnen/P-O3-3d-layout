# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# errors are fine, importing local files in ./pytorch_files/
from pytorch_files.engine import train_one_epoch, evaluate
import pytorch_files.utils as utils
import pytorch_files.transforms as T

import torchvision.models as models
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
        # target["masks"] = masks # TODO no masks!
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # load an instance segmentation model pre-trained on COCO
    #model = models.wide_resnet50_2()

    # get the number of input features for the classifier
    #in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    paths = ('./data/apart_0/', './data/meer_pers_0/', './data/zittend_0/', './data/apart_1/', './data/meer_pers_1/', './data/zittend_1/', './data/zz_testing/')

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print('Train on GPU.')
    else:
        print('Train on CPU.')
    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PO3Dataset(paths, get_transform(train=True))
    dataset_test = PO3Dataset(paths, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50]) # TODO: 50 laatste afbeeldingen als test? veel!
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = build_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 100 epochs, eta:
    num_epochs = 35

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
    torch.save(model, "training_PO3_v1.pth")


if __name__ == "__main__":
    main()
