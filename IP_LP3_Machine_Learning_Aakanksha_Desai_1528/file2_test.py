# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 23:41:56 2020

@author: aakan
"""

from keras.models import load_model
#from keras.models import Sequential
#import cv2
import numpy as np
#import os
import argparse
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required = True,help="path of image")
args = vars(ap.parse_args())
img_path = args["image"]
image = Image.open(img_path)
image = image.resize((64,64))
image = np.array(image)
if image.shape[2] == 4: 
    image = image[..., :3]
image = np.expand_dims(image, axis=0)
image = image/127.5
image = image - 1.0

label = ['RED','GREEN','BLUE']

model = load_model('model.h5')
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
#x = cv2.imread("C:\\Users\\aakan\\OneDrive\\Documents\\PROJECTS\\CC Assignment\\images\\test_img\\GREEN\\58.png")

#x = cv2.resize(x,(64,64))
#x = np.reshape(x,[1,64,64,3])
classes = model.predict_classes(image,batch_size=10)
#print(classes)
for i in classes:
    print(label[i])
