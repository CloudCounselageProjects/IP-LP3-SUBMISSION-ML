from PIL import Image
import random
import tensorflow as tf
# count={'1':0,'2':0,'3':0}
# for i in range(1000):
#     #r=1,g=2,b=3
#     dict={'1':random.randint(0,255),'2':random.randint(0,255),'3':random.randint(0,255) }
#     img = Image.new('RGB', (28,28), color = (dict['1'],dict['2'],dict['3']))
#     colour=max(dict, key=dict.get)
#     count[colour]+=1
#     #print(colour, dict)
#     img.save('Dataset/'+colour+'/'+str(count[colour])+'.png')

import cv2
import glob
import numpy as np

train_images = []
train_labels= []
for i in range(3):
    files = glob.glob ("C:/Users/aakas/Desktop/SampleProjects/ColorClassification/Dataset/"+str(i+1)+"/*.PNG")
    for myFile in files:
        #print(myFile)
        image = cv2.imread (myFile)
        train_images.append (image)
        train_labels.append (i+1)
train_images=np.array(train_images,dtype='float32')
train_labels=np.array(train_labels,dtype='float64')

test_images = []
test_labels= []
for i in range(3):
    files = glob.glob ("C:/Users/aakas/Desktop/SampleProjects/ColorClassification/Test/"+str(i+1)+"/*.PNG")
    for myFile in files:
        #print(myFile)
        image = cv2.imread (myFile)
        test_images.append (image)
        test_labels.append (i+1)
test_images=np.array(test_images,dtype='float32')
test_labels=np.array(test_labels,dtype='float64')
train_images=train_images/255.0
test_images=test_images/255.0


import matplotlib.pyplot as plt

gtrain_images=[]
for i in train_images:
    i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    gtrain_images.append(i)
gtrain_images=np.array(gtrain_images,dtype='float32')
gtest_images=[]
for i in test_images:
    i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    gtest_images.append(i)
gtest_images=np.array(gtest_images,dtype='float32')

gtrain_images=gtrain_images/255.0
gtest_images=gtest_images/255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28,3)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer= 'adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=10)

test_loss,test_acc=model.evaluate(test_images,test_labels)
print("test acc:",test_acc)

model.save("rgbcolourclassifiernormalized.h5")