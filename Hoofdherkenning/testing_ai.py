import cv2 as cv
import torch
from PIL import Image
import PO3_dataset
from math import floor
import time
import torchvision
import multiprocessing
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

"""
This file is used to test models.
This file is a draft which changes on what exactly should be extracted.
"""

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('Evaluate on GPU.')
else:
    print('Evaluate on CPU.')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
model.load_state_dict(torch.load("./saved_models/PO3_v4/training_49.pt"))
model.to(device)
model.eval()
torch.no_grad()

transform = PO3_dataset.get_transform(train=False)
def floor_lists(box):
    result = []
    for coordinates in box:
        result.append(list(map(lambda x: floor(float(x)), coordinates)))
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


def get_bboxes(img):
    img = Image.fromarray(img).convert("RGB")
    img, _ = transform(img, {})
    img = img.to(device)

    torch.cuda.synchronize()
    outputs = model([img])
    outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]
    return outputs[0]


print("extracting frames")
cap = cv.VideoCapture('./videos/output_more_person_1.avi')
fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('boxes_mor<²                                                           ²²      >_person_1.avi', fourcc, 10.0, (1920, 1080))
imgs = []
frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
count = 0
perc = 10

begintime=time.time()
total_frames = 50
cv.namedWindow("Results", cv.WINDOW_NORMAL)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('extracting frames finished')
        break

    # frame = cv.resize(frame, (floor(1920/4), floor(1080/4)))
    start = time.time()
    output = get_bboxes(frame)
    print("time: ", time.time()-start)

    boxes, scores, labels = floor_lists(output['boxes'].tolist()), output['scores'].tolist(), output['labels'].tolist()
    good_boxes, good_labels = [], []

    wait = False
    # filter based on score
    for i, score in enumerate(scores):
        if score > 0.9:
            good_boxes.append(boxes[i])
            good_labels.append(labels[i])

    for i, box in enumerate(good_boxes):
        if good_labels[i] == 1:
            cv.rectangle(frame, (box[0], box[1], box[2]-box[0], box[3]-box[1]), color=(0, 255, 0), thickness=2)
        else:
            cv.rectangle(frame, (box[0], box[1], box[2] - box[0], box[3] - box[1]), color=(0, 0, 255), thickness=2)
    out.write(frame)
    cv.imshow('Results', frame)
    cv.waitKey(1)
    count += 1

out.release()
print((time.time()-begintime)/total_frames)

