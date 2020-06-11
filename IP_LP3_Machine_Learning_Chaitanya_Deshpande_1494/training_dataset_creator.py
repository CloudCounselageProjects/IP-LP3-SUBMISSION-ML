import os
import random

import cv2
import numpy as np


def create_data(color):
    counter = 0
    tot = 0
    for i in range(25, 245, 10):
        counter += 1
        if color == 'red':
            temp[:, 0:width] = (0, 0, i)

        elif color == 'blue':
            temp[:, 0:width] = (i, 0, 0)

        elif color == 'green':
            temp[:, 0:width] = (0, i, 0)

        data_path = dir + color + "_" + str(counter) + '.jpg'

        cv2.imwrite(data_path, temp)
        for c1 in range(110, 230, 120):
            for c2 in range(110, 230, 120):
                tot += 1
                if color == 'red':
                    temp[:, 0:width] = (c1, c2, i)

                elif color == 'blue':
                    temp[:, 0:width] = (i, c1, c2)

                elif color == 'green':
                    temp[:, 0:width] = (c1, i, c2)

                data_path = dir + color + "_" + \
                    str(counter) + str(tot) + '.jpg'
                cv2.imwrite(data_path, temp)


def get_dataset(dir):
    total_images = os.listdir(dir)
    total_red = [x for x in total_images if x.split('_')[0] == 'red']
    total_blue = [x for x in total_images if x.split('_')[0] == 'blue']
    total_green = [x for x in total_images if x.split('_')[0] == 'green']

    print('total images:', len(total_images))
    print('total red:', len(total_red))
    print('total blue:', len(total_blue))
    print('total green:', len(total_green))

    return total_images


if __name__ == '__main__':
    height, width = 100, 100
    temp = np.zeros((height, width, 3), np.uint8)
    dir = 'training_dataset\\'
    create_data('blue')
    create_data('red')
    create_data('green')
    get_dataset(dir)
