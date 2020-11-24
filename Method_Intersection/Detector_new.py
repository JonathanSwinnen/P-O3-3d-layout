import torch
import torchvision
from PIL import Image
import helper_code.transforms as T
import cv2 as cv
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
from math import floor


def floor_lists(boxes):
    result = []
    for box in boxes:
        result.append(list(map(lambda x: floor(float(x)), box)))
    return result


class Detector:
    """
    Custom class for implementing self trained model.
    """

    def __init__(self, path_model="./model/training_23.pth"):
        # define device (gpu/cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Evaluate on GPU.') if torch.cuda.is_available() else print('No GPU available, evaluating on CPU.')

        # TODO load from dict
        # loading the model
        # self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        # in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # 2 = number_classes

        # self.model.load_state_dict(torch.load(path_model))

        # temp solution
        self.model = torch.load(path_model)

        # evaluation mode
        self.model.eval()
        torch.no_grad()

        # define transformation
        transforms = []
        transforms.append(T.ToTensor())
        self.transform = T.Compose(transforms)

    def detect_both_frames(self, left_frame, right_frame, min_score=0.9, filter_bad=True):
        # convert images to right format
        img_L = Image.fromarray(left_frame).convert("RGB")
        img_R = Image.fromarray(right_frame).convert("RGB")
        img_L, _ = self.transform(img_L, dict())
        img_R, _ = self.transform(img_R, dict())

        # imgs on gpu if available
        imgs = [img_L.to(self.device), img_R.to(self.device)]

        # evaluate images
        torch.cuda.synchronize()
        outputs = self.model(imgs)  # this takes the most time

        # extracting from
        output_L, output_R = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]
        boxes_L, scores_L, labels_L = (floor_lists(output_L["boxes"].tolist()),
                                       output_L["scores"].tolist(),
                                       output_L["labels"].tolist())

        boxes_R, scores_R, labels_R = (floor_lists(output_R["boxes"].tolist()),
                                       output_R["scores"].tolist(),
                                       output_R["labels"].tolist())

        # filter based on score
        good_boxes_L, good_labels_L, co_L, good_boxes_R, good_labels_R, co_R = [], [], [], [], [], []
        for i, score in enumerate(scores_L):
            if score > min_score:
                good_boxes_L.append(boxes_L[i])
                good_labels_L.append(labels_L[i])
                co_L.append(((boxes_L[i][0] + boxes_L[i][2])//2,
                             (boxes_L[i][1] + boxes_L[i][3])//2))

        for i, score in enumerate(scores_R):
            if score > min_score:
                good_boxes_R.append(boxes_R[i])
                good_labels_R.append(labels_R[i])
                co_R.append(((boxes_R[i][0] + boxes_R[i][2])//2,
                             (boxes_R[i][1] + boxes_R[i][3])//2))
        print(co_L, co_R, good_boxes_L, good_boxes_R)
        return co_L, co_R, good_boxes_L, good_boxes_R
