# -*- coding: utf-8 -*-
"""
@author: MAYANK
"""
#importing libraries
import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
#saving directory
img_dir = 'images'
print(os.listdir(img_dir))


#creating empty varibles
x = []
y = []
dataset = []

#creating dataset
for i in os.listdir(img_dir):
    path = os.path.join(img_dir,i)
    for im in os.scandir(path):
        img = cv2.imread(im.path)
        img = cv2.resize(img,(224,224))
        dataset.append([img,i])

#shuffling dataset
random.shuffle(dataset)

#fillingfetures and labels in the variables
for features,labels in dataset:
    x.append(features)
    y.append(labels)

#converting categories into numerical values {0,1,2}
from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
y = l.fit_transform(y)


x = np.array(x)
y = y.reshape(y.shape[0],1)

# print shape
print(x.shape)
print(y.shape)

#Normalizing the dataset
x = x/255.0
print(x[0])
print(x.shape)

model = Sequential([Conv2D(64,(3,3),input_shape=(224,224,3),activation='relu'),
                    MaxPooling2D((2,2)),
                    Dropout(0.2),
                    Conv2D(64,(3,3),activation='relu'),
                    MaxPooling2D((2,2)),
                    Dropout(0.2),
                    Conv2D(64,(3,3),activation='relu'),
                    MaxPooling2D((2,2)),
                    Dropout(0.2),
                    Flatten(),
                    Dense(64,activation='relu'),
                    Dense(3,activation='softmax')
                   ])

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(x,y,batch_size=32,epochs=10,validation_split=0.3)
print(model.summary())
model.save('train_model.h5')

print(model.metrics_names)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel(['epochs'])
plt.ylabel(['accuracy'])
plt.title('Accuracy Graph')
plt.legend(['accuracy','val_accuracy'])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel(['epochs'])
plt.ylabel(['loss'])
plt.title('Loss Graph')
plt.legend(['loss','val_loss'])
