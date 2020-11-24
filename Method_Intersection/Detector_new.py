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


def calc_area(box):
    return (box[2]-box[0])*(box[3]-box[1])



def iou(box1, box2):
    iou_b = [max(box1[0], box2[0]),
           max(box1[1], box2[1]),
           min(box1[2], box2[2]),
           min(box1[3], box2[3])]
    if iou_b[2]<iou_b[0] or iou_b[3]<iou_b[1]:
        return 0
    iou_ar = calc_area(iou_b)
    return iou_ar / min(calc_area(box1), calc_area(box2))


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
        self.transform = T.Compose([T.ToTensor()])

    def detect_both_frames(self, left_frame, right_frame, min_score=0.9, filter_doubles=True):
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
                add_im = True
                if filter_doubles:  # filter overlapping boxes
                    for j in range(len(good_boxes_L)):
                        if iou(boxes_L[i], good_boxes_L[j]) > 0.8:
                            if calc_area(boxes_L[i]) < calc_area(good_boxes_L[j]):
                                del good_boxes_L[j]
                            else:
                                add_im = False
                                break
                if add_im:
                    good_boxes_L.append(boxes_L[i])
                    good_labels_L.append(labels_L[i])
                    co_L.append(((boxes_L[i][0] + boxes_L[i][2])//2,
                                 (boxes_L[i][1] + boxes_L[i][3])//2))

        for i, score in enumerate(scores_R):
            if score > min_score:
                add_im = True
                if filter_doubles:  # filter overlapping boxes
                    for j in range(len(good_boxes_R)):
                        # if overlapping for > 80%, only add the smallest box
                        if iou(boxes_R[i], good_boxes_R[j]) > 0.8:
                            if calc_area(boxes_R[i]) < calc_area(good_boxes_R[j]):
                                del good_boxes_R[j]
                            else:
                                add_im = False
                                break
                if add_im:
                    good_boxes_R.append(boxes_R[i])
                    good_labels_R.append(labels_R[i])
                    co_R.append(((boxes_R[i][0] + boxes_R[i][2])//2,
                                 (boxes_R[i][1] + boxes_R[i][3])//2))

        return co_L, co_R, good_boxes_L, good_boxes_R
