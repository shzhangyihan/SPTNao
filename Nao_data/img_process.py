import numpy as np 
import cv2
import csv

# hyperparameters
canny_lower = 100
canny_upper = 250
resize_x = 40
resize_y = 30

img_path = "./img_data/camImage0.png"
img = cv2.imread(img_path, 0)
iterator = 0

csv_array = []
while not img is None:
    # img processing
    print "Start processing img " + str(iterator)
    img = cv2.resize(img, (resize_x, resize_y))
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(blur, canny_lower, canny_upper)
    img_edge_path = "./edge_data/camImage" + str(iterator) + ".png"
    cv2.imwrite(img_edge_path, canny)
    csv_array.append(canny)
    print canny.shape
    print "Processing success for img " + str(iterator)
    # retrieve new img
    iterator += 1
    img_path = "./img_data/camImage" + str(iterator) + ".png"
    img = cv2.imread(img_path, 0)

print "Finish processing for " + str(iterator) + " images"
f = open('img_process.csv', 'w')
writer = csv.writer(f, lineterminator = '\n')
writer.writerows(csv_array)
f.close()
print np.asarray(csv_array).shape
