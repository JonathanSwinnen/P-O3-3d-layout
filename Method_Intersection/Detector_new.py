import torch
import torchvision
from PIL import Image
import helper_code.transforms as T
import cv2 as cv
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
from math import floor
import pickle


def center_of_box(box):
    return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2


def floor_lists(boxes):
    result = []
    for box in boxes:
        result.append(list(map(lambda x: floor(float(x)), box)))
    return result


def calc_area(box):
    return (box[2]-box[0])*(box[3]-box[1])



def adjusted_iou(box1, box2):
    iou_b = [max(box1[0], box2[0]),
           max(box1[1], box2[1]),
           min(box1[2], box2[2]),
           min(box1[3], box2[3])]
    if iou_b[2]<iou_b[0] or iou_b[3]<iou_b[1]:
        return 0
    iou_ar = calc_area(iou_b)
    return iou_ar / min(calc_area(box1), calc_area(box2))


def split_arrays(boxes, scores, labels):
    boxes_1, boxes_2, scores_1, scores_2 = [], [], [], []

    for i, label in enumerate(labels):
        if label == 1:
            boxes_1.append(boxes[i])
            scores_1.append(scores[i])
        elif label == 2:
            boxes_2.append(boxes[i])
            scores_2.append(scores[i])
    return boxes_1, scores_1, boxes_2, scores_2


def main_filter(boxes, scores, min_score):
    # output vars
    good_boxes, co = [], []

    # filter
    for i, box in enumerate(boxes):
        if scores[i] > min_score:
            add_im = True

            # setup while loop
            # deleting images in loop from array doesn't allow for a forloop
            j = 0
            run = True
            if len(good_boxes) == 0:
                run = False

            while run:
                if adjusted_iou(box, good_boxes[j]) > 0.8:
                    if calc_area(box) < calc_area(good_boxes[j]):
                        del good_boxes[j]
                        del co[j]
                        j -= 1
                    else:
                        add_im = False
                        break
                j += 1
                if len(good_boxes) == j:
                    run = False

            if add_im:
                good_boxes.append(box)
                co.append(center_of_box(box))
    return good_boxes, co


class Detector:
    """
    Custom class for implementing self trained model.
    """

    def __init__(self, path_model="./data/model/training_49_full.pt"):
        # define device (gpu/cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Evaluate on GPU.') if torch.cuda.is_available() else print('No GPU available, evaluating on CPU.')

        # loading model from dict
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
        self.model.load_state_dict(torch.load(path_model))
        self.model.to(self.device)

        # OLD (can produce multiple errors!)
        # self.model = torch.load(path_model)

        # evaluation mode
        self.model.eval()
        torch.no_grad()

        # define transformation
        self.transform = T.Compose([T.ToTensor()])

    def detect_both_frames(self, left_frame, right_frame, min_score_h=0.9, min_score_m=0.9):
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

        # split heads and masks
        boxes_L_h, scores_L_h, boxes_L_m, scores_L_m = split_arrays(boxes_L, scores_L, labels_L)
        boxes_R_h, scores_R_h, boxes_R_m, scores_R_m = split_arrays(boxes_R, scores_R, labels_R)

        # filter
        g_boxes_L_h, co_L_h = main_filter(boxes_L_h, scores_L_h, min_score_h)
        g_boxes_L_m, co_L_m = main_filter(boxes_L_m, scores_L_m, min_score_m)

        g_boxes_R_h, co_R_h = main_filter(boxes_R_h, scores_R_h, min_score_h)
        g_boxes_R_m, co_R_m = main_filter(boxes_R_m, scores_R_m, min_score_m)


        return [(co_L_h, co_R_h, g_boxes_L_h, g_boxes_R_h),
                (co_L_m, co_R_m, g_boxes_L_m, g_boxes_R_m)]


if __name__ == "__main__":
    print("Extracting data from video and saving it.")
    cap_1 = cv.VideoCapture('./data/videos/output_more_person_1.avi')
    cap_2 = cv.VideoCapture('./data/videos/output_more_person_0.avi')

    detector = Detector()

    all_info = []
    total_frames = int(cap_1.get(cv.CAP_PROP_FRAME_COUNT))
    print("Frames:", total_frames)
    print("Est. time in min:", total_frames*0.57/60)

    start = time.time()
    i = 0
    while cap_1.isOpened():
        ret_1, frame_1 = cap_1.read()
        if not ret_1:
            print("failed to grab frame_1")
            break

        ret_2, frame_2 = cap_2.read()
        if not ret_2:
            print("failed to grab frame_2")
            break
        # frame_1, frame_2 = cv.resize(frame_1, (480, 270)), cv.resize(frame_2, (480, 270))
        data = detector.detect_both_frames(frame_1, frame_2)
        boxes_h = data[0][2]
        boxes_m = data[1][2]

        all_info.append(detector.detect_both_frames(frame_1, frame_2))
        i += 1
        if i%10 == 0:
            print(i/total_frames*100)
    end = time.time()
    print("Total time:", end-start)
    print("Total frames:", len(all_info))
    print("Time/frame:", (end-start)/len(all_info))

    print(all_info)
    pickle_out = open("data/video_data/more_person_new.pckl", "wb")
    pickle.dump(all_info, pickle_out)
    pickle_out.close()

    cap_1.release()
    cap_2.release()