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
eight_bit_path = {
    0: './8bit_img/45/',
    1: './8bit_img/90/',
    2: './8bit_img/135/',
    3: './8bit_img/180/',
    4: './8bit_img/225/',
    5: './8bit_img/270/',
    6: './8bit_img/315/',
    7: './8bit_img/self/',
}
sub_path = {
    0: 'both/',
    1: 'left/',
    2: 'right/',
}

for i in range(8):
    for j in range(3):
        dir_path = root_path[i] + sub_path[j]
        dir_edge_path = edge_path[i] + sub_path[j]
        dir_four_path = four_bit_path[i] + sub_path[j]
        dir_eight_path = eight_bit_path[i] + sub_path[j]
        
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
        start = 1
        while not img is None:
            # process edge img
            canny = cv2.Canny(img, canny_lower, canny_upper)
            img_edge_path = dir_edge_path + 'img_' + str(start) + '.png'
            cv2.imwrite(img_edge_path, canny)
            
            # process 4 bit img
            black_indices = img < 110
            four = img
            four[black_indices] = 0
            four = np.divide(img, 64)
            four = np.multiply(four, 64)
            img_four_path = dir_four_path + 'img_' + str(start) + '.png'
            cv2.imwrite(img_four_path, four)
            
            # process 8 bit img
            eight = img
            eight[black_indices] = 0
            eight = np.divide(img, 32)
            eight = np.multiply(eight, 32)
            img_eight_path = dir_eight_path + 'img_' + str(start) + '.png'
            cv2.imwrite(img_eight_path, eight)
            
            # retrive next img
            iterator += 1
            start += 1
            img_path = dir_path + 'img_' + str(iterator) + '.png'
            img = cv2.imread(img_path, 0)
        print 'End: ', iterator - 1
