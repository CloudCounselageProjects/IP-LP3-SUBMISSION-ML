import numpy as np
from PIL import Image
import random

def DataGenerator(arr):
    if len(arr)==0:
        print("Invalid Input")
        return
    for color in arr:
        array = np.zeros([200,200,3], dtype=np.uint8)
        array[:,:] = [ color[0], color[1], color[2] ]

        img = Image.fromarray(array)
        img.save('./custom_images/custom_img_{}.png'.format(random.randint(10,99)))
        print(color,"image got produced")


'''
    input:
        arr = [ [r,g,b] ]           ## where r,g,b => {0,255} 
    example:
        arr = [ [0,255,0], [255,0,0], [0,0,255]] 
        Datagenerator(arr)
        
        Above example will generate 3 images of respective rgb colours given as input.
'''
        
arr = [ [255,102,204] ]

DataGenerator(arr)
        

