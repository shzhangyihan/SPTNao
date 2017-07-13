import numpy as np
import cv2

# hyperparameters
canny_lower = 100
canny_upper = 250

# data dir path
root_path = {
    0: './raw_img/45/',
    1: './raw_img/90/',
    2: './raw_img/135/',
    3: './raw_img/180/',
    4: './raw_img/225/',
    5: './raw_img/270/',
    6: './raw_img/315/',
    7: './raw_img/self/',
}
edge_path = {
    0: './edge_img/45/',
    1: './edge_img/90/',
    2: './edge_img/135/',
    3: './edge_img/180/',
    4: './edge_img/225/',
    5: './edge_img/270/',
    6: './edge_img/315/',
    7: './edge_img/self/',
}
four_bit_path = {
    0: './4bit_img/45/',
    1: './4bit_img/90/',
    2: './4bit_img/135/',
    3: './4bit_img/180/',
    4: './4bit_img/225/',
    5: './4bit_img/270/',
    6: './4bit_img/315/',
    7: './4bit_img/self/',
}
sub_path = {
    0: 'both/',
    1: 'left/',
    2: 'right/',
}

for i in range(8):
    for j in range(3):
        dir_path = root_path[i] + sub_path[j]
        edge_path = edge_path[i] + sub_path[j]
        four_bit_path = four_bit_path[i] + sub_path[j]
        print "Start processing dir: " + dir_path
        iterator = 0
        img_path = dir_path + 'img_' + str(iterator) + '.png'
        img = cv2.imread(img_path, 0)
        # find start point
        while img is None:
            iterator += 1
            img_path = dir_path + 'img_' + str(iterator) + '.png'
            img = cv2.imread(img_path, 0)
        print 'Start: ', iterator
        start = 0
        while not img is None:
            start += 1
            iterator += 1
            img_path = dir_path + 'img_' + str(iterator) + '.png'
            img = cv2.imread(img_path, 0)
            canny = cv2.Canny(img, canny_lower, canny_upper)
            img_edge_path = edge_path + 'img_' + str(start) + '.png'
            cv2.imwrite(img_edge_path, canny)
        print iterator
