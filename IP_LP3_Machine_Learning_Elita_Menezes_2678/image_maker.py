#Approach 2
#Created a dataset using opencv2 and numpy array
#Advantage : A collection of images of a single color(completely random)
#            Fast images generated


import numpy as np
import cv2
import random

def create_image(height, width, color, image_no):
    #create a 3 channel np.array
    image = np.zeros((height, width, 3), np.uint8)
    #fill array with the respective color
    image[:] = color
    #save image in train or validation set
    cv2.imwrite('testing/' + image_no +'.jpg', image)


#for training and validation
# for i in range(50):
#     if i % 2 == 0:
#     #randomly choose color of image
#         r = random.randrange(255)
#         g = random.randrange(16)
#         b = random.randrange(16)
        
#     else:
#         r = random.randrange(220, 255)
#         g = random.randrange(100)
#         b = random.randrange(100)

#     create_image(224, 224, (b, g, r), str(i))

#for testing images
for i in range(1, 10):
    r = random.randrange(255)
    g = random.randrange(255)
    b = random.randrange(255)
    print(f'For image {i} : Red - {r} Green - {g} Blue - {b}')
    create_image(244, 244, (b, g, r), str(i))
