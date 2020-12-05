import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from pytorch_files.coco_utils import get_coco_api_from_dataset
from pytorch_files.coco_eval import CocoEvaluator
import pytorch_files.utils as utils
from math import floor, sqrt


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
    if iou_b[2] < iou_b[0] or iou_b[3] < iou_b[1]:
        return 0
    iou_ar = calc_area(iou_b)
    return iou_ar / (calc_area(box1) + calc_area(box1) - iou_ar)


def adjusted_iou(box1, box2):
    iou_b = [max(box1[0], box2[0]),
           max(box1[1], box2[1]),
           min(box1[2], box2[2]),
           min(box1[3], box2[3])]
    if iou_b[2] < iou_b[0] or iou_b[3] < iou_b[1]:
        return 0
    iou_ar = calc_area(iou_b)
    return iou_ar / min(calc_area(box1), calc_area(box2))


def center_of_box(box):
    return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2


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


def main_filter(boxes, scores):
    # vars
    score_distribution = dict()
    good_boxes, co = [], []
    filtered_score, filtered_iou = 0, 0

    # filter (+- same method as in Detector_new.py)
    for i, box in enumerate(boxes):
        rounded_score = round(scores[i], 3)
        score_distribution[rounded_score] = score_distribution.get(rounded_score, 0) + 1
        if scores[i] > 0.9:
            add_im = True
            j = 0
            run = True
            if len(good_boxes) == 0:
                run = False
            while run:
                if adjusted_iou(box, good_boxes[j]) > 0.8:
                    filtered_iou += 1
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
        else:
            filtered_score += 1
    return score_distribution, good_boxes, co, filtered_score, filtered_iou


def sum_score_distributions(score1, score2):
    for k, v in score2.items():
        score1[k] = score1.get(k, 0) + v
    return score1


def norm(co1, co2):
    return sqrt((co1[0]-co2[0])**2 + (co1[1]-co2[1])**2)


def pair_boxes(boxes_target, co_target, boxes_predicted, co_predicted):
    recognised, wrong = 0, 0
    paired_boxes = dict()
    for i, box in enumerate(boxes_target):
        paired_boxes[i] = [(box, co_target[i])]

    for i, co in enumerate(co_predicted):
        smallest_dist, smallest_id = float("inf"), -1
        for id, target in paired_boxes.items():
            dist = norm(target[0][1], co)
            if dist < smallest_dist:
                smallest_dist = dist
                smallest_id = id
        if smallest_id == -1:
            wrong += 1
        else:
            if len(paired_boxes[smallest_id]) == 1:
                if iou(paired_boxes[smallest_id][0][0], boxes_predicted[i]) > 0:
                    recognised += 1
                    paired_boxes[smallest_id] = [paired_boxes[smallest_id][0], (boxes_predicted[i], co), smallest_dist]
                else:
                    wrong += 1
            else:
                wrong += 1
                if smallest_dist < paired_boxes[smallest_id][2]:
                    paired_boxes[smallest_id] = [paired_boxes[smallest_id][0], (boxes_predicted[i], co), smallest_dist]

    all_ious, all_dists = [], []

    for paired in paired_boxes.values():
        if len(paired) > 1:
            all_ious.append(iou(paired[0][0], paired[1][0]))
            all_dists.append(paired[2])

    average_iou, average_dist = 0, 0
    if len(all_ious) > 0:
        average_iou = sum(all_ious)/len(all_ious)
        average_dist = sum(all_dists)/len(all_dists)

    return len(paired_boxes.keys()), recognised, wrong, average_iou, average_dist


def calc_score(targets, output):
    if len(output) != 1:
        raise Exception('More then 1 output. Custom evaluation requires just one output.')

    boxes, scores, labels = (floor_lists(output[0]["boxes"].tolist()),
                                output[0]["scores"].tolist(),
                                output[0]["labels"].tolist())
    heads, heads_score, masks, masks_score = split_arrays(boxes, scores, labels)

    score_distr_heads, good_heads, co_heads, f_score_heads, f_iou_heads = main_filter(heads, heads_score)
    score_distr_masks, good_masks, co_masks, f_score_masks, f_iou_masks = main_filter(masks, masks_score)

    # filtered_score = f_score_heads + f_score_masks
    # filtered_iou = f_iou_heads + f_iou_masks
    # score_distribution = sum_score_distributions(score_distr_heads, score_distr_masks)

    target_boxes, target_labels = floor_lists(targets[0]["boxes"].tolist()), targets[0]["labels"].tolist()
    target_coordinates = []
    for box in target_boxes:
        target_coordinates.append(center_of_box(box))

    target_heads, target_co_heads, target_masks, target_co_masks = [], [], [], []
    for i, label in enumerate(target_labels):
        if label == 1:
            target_heads.append(target_boxes[i])
            target_co_heads.append(target_coordinates[i])
        elif label == 2:
            target_masks.append(target_boxes[i])
            target_co_masks.append(target_coordinates[i])

    nb_heads, nb_recognized_heads, nb_incorrect_heads, matched_area_heads, dist_heads = \
        pair_boxes(target_heads, target_co_heads, good_heads, co_heads)

    nb_masks, nb_recognized_masks, nb_incorrect_masks, matched_area_masks, dist_masks = \
        pair_boxes(target_masks, target_co_masks, good_masks, co_masks)

    return [(score_distr_heads, f_score_heads, f_iou_heads, nb_heads, nb_recognized_heads, nb_incorrect_heads, matched_area_heads, dist_heads),
            (score_distr_masks, f_score_masks, f_iou_masks, nb_masks, nb_recognized_masks, nb_incorrect_masks, matched_area_masks, dist_masks)]






def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    score_dist_h, score_dist_m = dict(), dict()

    time_dist = []
    data = []
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        time_dist.append(model_time)
        im_data = calc_score(targets, outputs)

        score_dist_h = sum_score_distributions(score_dist_h, im_data[0][0])
        score_dist_m = sum_score_distributions(score_dist_m, im_data[1][0])

        data.append((im_data[0][1:], im_data[1][1:]))

    # gather the stats from all processes

    metric_logger.synchronize_between_processes()
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    f_score_h, f_iou_h, nb_h, g_h, f_h, iou_h, dist_h = 0, 0, 0, 0, 0, 0, 0
    f_score_m, f_iou_m, nb_m, g_m, f_m, iou_m, dist_m = 0, 0, 0, 0, 0, 0, 0
    print(data)
    for it in data:
        f_score_h += it[0][0]
        f_iou_h += it[0][1]
        nb_h += it[0][2]
        g_h += it[0][3]
        f_h += it[0][4]
        iou_h += it[0][3] * it[0][5]
        dist_h += it[0][3] * it[0][6]

        f_score_m += it[1][0]
        f_iou_m += it[1][1]
        nb_m += it[1][2]
        g_m += it[1][3]
        f_m += it[1][4]
        iou_m += it[1][3] * it[1][5]
        dist_m += it[1][3] * it[1][6]

    iou_h = iou_h / g_h
    dist_h = dist_h / g_h
    if g_m != 0:
        iou_m = iou_m / g_m
        dist_m = dist_m / g_m

    data_test = (time_dist, (score_dist_h, f_score_h, f_iou_h, nb_h, g_h, f_h, iou_h, dist_h),
                 (score_dist_m, f_score_m, f_iou_m, nb_m, g_m, f_m, iou_m, dist_m))

    return coco_evaluator, data_test
