# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 02:09:59 2020

@author: aakan
"""

import random
import numpy as np
import cv2
from PIL import Image
#img = np.zeros((64,64,3),dtype="uint8")
count = 0
for i in range(100):
    b=random.randint(0,255)
    g=random.randint(0,255)
    r=random.randint(0,255)
    color = (r,g,b)

    img = Image.new(mode = "RGB",size = (64,64),color = color)
    count += 1
        #img[:] = color
    print(r,g,b)
        #cv2.imwrite('1.jpg',img)
    m = max(r,g,b)
    index = color.index(m)
    #img.show()
    img.save('C:\\Users\\aakan\OneDrive\Documents\PROJECTS\CC Assignment\\'+str(index)+'\\'+str(count)+'.jpg')
    
