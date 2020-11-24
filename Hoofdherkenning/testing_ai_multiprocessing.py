import cv2 as cv
import torch
from PIL import Image
import PO3_dataset
import time
import multiprocessing




def get_bboxes(im, T, m, d):
    im = Image.fromarray(im).convert("RGB")
    im, _ = T(im, {})
    im = im.to(d)

    # torch.cuda.synchronize()
    outputs = m([im])
    outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]
    return outputs[0]


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.is_available():
        print('Evaluate on GPU.')
    else:
        print('Evaluate on CPU.')

    model = torch.load("./saved_models/PO3_v3/training_23.pth")  # , map_location=torch.device("cpu"))
    model.eval()
    torch.no_grad()

    transform = PO3_dataset.get_transform(train=False)

    cap = cv.VideoCapture('./videos/output_TAFELS_1.avi')
    imgs = []
    i = 1

    print("start extracting frames")
    while cap.isOpened():
        ret, frame = cap.read()
        imgs.append(frame)
        if i == 100:
            break
        i += 1
    print("finished reading frames")
    start_tot = time.time()
    for i in range(50):
        start = time.time()
        get_bboxes(imgs[i], transform, model, device)
        print(time.time()-start)

    #processes = []
    #for i in range(5):
    #    p = multiprocessing.Process(target=get_bboxes, args=(imgs[i], transform, model, device))
    #    processes.append(p)
    #    p.start()
#
    #for process in processes:
    #    process.join()
    print(time.time() - start_tot)