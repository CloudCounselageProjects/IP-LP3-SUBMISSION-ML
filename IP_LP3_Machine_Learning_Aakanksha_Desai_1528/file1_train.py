# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 23:22:56 2020

@author: aakan
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten
from keras.models import Sequential
model = Sequential([
                 Conv2D(32,(3,3),input_shape=(64,64,3)),
                Activation('relu'),
    
                MaxPooling2D(2,2),
    
                Conv2D(32,(3,3)),
                Activation('relu'),
                
                MaxPooling2D(2,2),
    
                Flatten(),
    
                Dense(units=128),
                Activation('relu'),
    
                Dense(units=3),
                Activation('sigmoid')
])

model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(r'C:\Users\aakan\OneDrive\Documents\PROJECTS\CC Assignment\new_train',target_size = (64,64),batch_size = 10,class_mode = 'categorical')
test_set = test_datagen.flow_from_directory(r'C:\Users\aakan\OneDrive\Documents\PROJECTS\CC Assignment\new_test',target_size = (64,64),batch_size = 10,class_mode = 'categorical')
#model.summary()
train_size = 500
val_size = 100
batch_size = 10
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
history = model.fit_generator(training_set,nb_epoch = 50, steps_per_epoch = train_size//batch_size,validation_data = test_set,validation_steps = val_size//batch_size)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
print('acc:',sum(acc)/len(acc)*100,'%')
print('val_acc:',sum(val_acc)/len(val_acc)*100,'%')
model.save("model.h5")
