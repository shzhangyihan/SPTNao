import numpy as np 
import cv2

img_path = "./img_data/camImage0.png"
img = cv2.imread(img_path, 0)
iterator = 0

while not img is None:
    # img processing
    print "Start processing img " + str(iterator)
    canny = cv2.Canny(img, 180, 250)
    img_edge_path = "./edge_data/camImage" + str(iterator) + ".png"
    cv2.imwrite(img_edge_path, canny)
    print "Processing success for img " + str(iterator)
    # retrieve new img
    iterator += 1
    img_path = "./img_data/camImage" + str(iterator) + ".png"
    img = cv2.imread(img_path, 0)

print "Finish processing for " + str(iterator) + " images"

