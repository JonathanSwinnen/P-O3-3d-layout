import cv2 as cv
import torch
from PIL import Image
import PO3_dataset
from math import floor
import time
import torchvision
import multiprocessing
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('Evaluate on GPU.')
else:
    print('Evaluate on CPU.')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
model.load_state_dict(torch.load("./saved_models/PO3_v5/training_49.pt"))
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
imgs = []
frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
count = 0
perc = 10

begintime=time.time()
total_frames = 50

cv.namedWindow("Results")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('extracting frames finished')
        break

    frame = cv.resize(frame, (floor(1920/4), floor(1080/4)))
    start = time.time()
    output = get_bboxes(frame)
    print("time: ", time.time()-start)

    boxes, scores, labels = floor_lists(output['boxes'].tolist()), output['scores'].tolist(), output['labels'].tolist()
    good_boxes, good_labels = [], []
    print(scores, labels)

    wait = False
    # filter based on score
    for i, score in enumerate(scores):
        if score > 0.9:
            add_im = True
            if False:  # filter overlapping boxes
                for j in range(len(good_boxes)):
                    if iou(boxes[i], good_boxes[j]) > 0.8:
                        if calc_area(boxes[i]) < calc_area(good_boxes[j]):
                            wait = True
                            cv.rectangle(frame,
                                        (    good_boxes[j][0],
                                             good_boxes[j][1],
                                             good_boxes[j][2] - good_boxes[j][0],
                                             good_boxes[j][3] - good_boxes[j][1]
                                        ),
                                         color=(255, 0, 0),
                                         thickness=2)
                            del good_boxes[j]
                        else:
                            cv.rectangle(frame,
                                         (boxes[i][0],
                                          boxes[i][1],
                                          boxes[i][2] - boxes[i][0],
                                          boxes[i][3] - boxes[i][1]
                                          ),
                                         color=(255, 0, 0),
                                         thickness=2)
                            wait = True
                            add_im = False
                            break
            if add_im:
                good_boxes.append(boxes[i])
                good_labels.append(labels[i])

    for i, score in enumerate(scores):
        if score > 0.9 and not any(iou(boxes[i], good_boxes[j]) > 0.8 for j in range(len(good_boxes))):
            good_boxes.append(boxes[i])
            good_labels.append(labels[i])

    for box in good_boxes:
        cv.rectangle(frame, (box[0], box[1], box[2]-box[0], box[3]-box[1]), color=(0, 255, 0), thickness=2)
    cv.imshow("Results", frame)
    if wait:
        print(count)
        cv.waitKey(1500)
    else:
        cv.waitKey(1)

    if perc/100 < count/frame_count:
        print(perc, "%")
        perc += 10
    count += 1

print((time.time()-begintime)/total_frames)

