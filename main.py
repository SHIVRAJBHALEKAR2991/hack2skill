from ultralytics import YOLO
import cv2 as cv
import math

model =YOLO("crater.pt")

classnames =['Crater']

img = cv.imread('base_file/valid/images/1_jpg.rf.0666e8c8eb5e13993eb61573d42cc3ec.jpg')
# cv.imshow('image',img)
# cv.waitKey(0)
result=model(img,stream=True)
for r in result:

    boxes=r.boxes
    for box in boxes:

        x1,y1,x2,y2 = box.xyxy[0]
        x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
        # class
        cls = box.cls[0]
        cv.rectangle(img,(x1,y1),(x2,y2),(0,0,255),thickness=2)
    cv.imshow('image',img)
    cv.waitKey(0)
