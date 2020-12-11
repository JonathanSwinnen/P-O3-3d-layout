import cv2
import pickle

img = cv2.imread("./combined_data/TA/im_31.png")
f = open('./combined_data/clean_ann_combined.pkl', 'rb')
ann = pickle.load(f)
f.close()

cv2.namedWindow("Results", cv2.WINDOW_NORMAL)
print(ann)
boxes = ann['./combined_data/TA/im_31.png']
for box in boxes[0]:
    print(box)
    cv2.rectangle(img, (box[0], box[1], box[2]-box[0], box[3]-box[1]), color=(0, 255, 0), thickness=2)
cv2.imshow("Results", img)
cv2.waitKey()