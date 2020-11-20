import cv2 as cv
import torch
import PO3_dataset



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
    i += 1


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    print('Train on GPU.')
else:
    print('Train on CPU.')

model = torch.load("training_PO3.pth")
model.eval()

torch.cuda.synchronize()
outputs = model(imgs)

outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]

print(outputs)

