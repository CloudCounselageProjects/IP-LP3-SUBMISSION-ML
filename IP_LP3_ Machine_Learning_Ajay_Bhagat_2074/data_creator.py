import os
import random

import cv2
import numpy as np


def create_img(color, add_other=False):
    counter = 0

    for ind in range(25, 245, 10):
        counter += 1
        if color == 'red':
            blank_image[:, 0:width] = (0, 0, ind)

        elif color == 'blue':
            blank_image[:, 0:width] = (ind, 0, 0)

        elif color == 'green':
            blank_image[:, 0:width] = (0, ind, 0)

        path_image = base + color + "_" + str(counter) + '.jpg'
        cv2.imwrite(path_image, blank_image)

        if add_other:
            tot = 0
            for c1 in range(110, 230, 120):
                for c2 in range(110, 230, 120):
                    tot += 1
                    if color == 'red':
                        blank_image[:, 0:width] = (c1, c2, ind)

                    elif color == 'blue':
                        blank_image[:, 0:width] = (ind, c1, c2)

                    elif color == 'green':
                        blank_image[:, 0:width] = (c1, ind, c2)

                    path_image = base + color + "_" + str(counter) + str(tot) + '.jpg'
                    cv2.imwrite(path_image, blank_image)


def create_custom(value):
    # BGR
    blank_image[:, 0:width] = value
    blue, green, red = value
    #    path_image = base + "cc_" + str(value) + '.jpg'
    path_image = base + "cc_(" + str(red) + ", " + str(green) + ", " + str(blue) + ').jpg'
    cv2.imwrite(path_image, blank_image)
    cv2.waitKey()


def get_dataset(base):
    total_images = os.listdir(base)
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
    blank_image = np.zeros((height, width, 3), np.uint8)

    # 0: train set, 1:test set
    set_type = 0

    if set_type == 0:
        base = 'dataset\\'
        add = True
        create_img('blue', add_other=add)
        create_img('red', add_other=add)
        create_img('green', add_other=add)
        get_dataset(base)

    if set_type == 1:
        base = 'input\\'
        for i in range(0, 100):
            create_custom((random.choice(range(0, 255)),
                           random.choice(range(0, 255)),
                           random.choice(range(0, 255))))
