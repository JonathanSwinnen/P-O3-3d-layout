import cv2 as cv
import torch
import pytorch_files.transforms as T


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

print("extracting frames")
cap = cv.VideoCapture('output_two_person_1.avi')
imgs = []
frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
i = 0
perc = 10

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or i == 1:
        print('extracting frames finished')
        break
    imgs.append(frame)
    if perc/100 < i/frame_count:
        print(perc, "%")
        perc += 10
    i+=1

#imgs += [cv.imread("./data/videodata_0/img/im_17.png")]

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    print('Train on GPU.')
else:
    print('Train on CPU.')


imgs, target = get_transform(train=False)(imgs[0], {})
imgs = list(img.to(device) for img in imgs)

model = torch.load("training_PO3.pth")
model.eval()

torch.cuda.synchronize()
outputs = model(imgs)

outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]


