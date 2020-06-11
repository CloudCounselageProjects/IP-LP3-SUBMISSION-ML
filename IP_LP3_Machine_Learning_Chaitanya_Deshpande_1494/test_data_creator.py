import os
import random
import cv2
import numpy as np


def create_test_data(value):
    temp[:, 0:width] = value
    blue, green, red = value
    test_image = dir + str(red) + "_" + \
        str(green) + "_" + str(blue) + '.jpg'
    cv2.imwrite(test_image, temp)
    cv2.waitKey()


if __name__ == '__main__':
    height, width = 100, 100
    temp = np.zeros((height, width, 3), np.uint8)
    dir = 'test_images\\'
    for i in range(0, 100):
        create_test_data((random.choice(range(0, 255)),
                          random.choice(range(0, 255)),
                          random.choice(range(0, 255))))
