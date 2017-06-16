import numpy as np
import cv2
import skvideo.io

# hyperparameters
canny_lower = 100
canny_upper = 250
resize_x = 40
resize_y = 30

video_path = "./1.mp4"
cap = skvideo.io.VideoCapture(video_path)
print cap.isOpened()

def img_process(frame, path):
    img = cv2.resize(frame, (resize_x, resize_y))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    canny = cv2.Canny(img, canny_lower, canny_upper)
    cv2.imwrite(path, canny)

iterator = 0
while(cap.isOpened()):
    print "Start reading frame " + str(iterator)
    ret, frame = cap.read()
    if ret == False:
        break
    img_path = "./video_data/camImage" + str(iterator/15) + ".png"
    if iterator % 18 == 0:
        img_process(frame, img_path)
    iterator += 1

print "Convert finished"
