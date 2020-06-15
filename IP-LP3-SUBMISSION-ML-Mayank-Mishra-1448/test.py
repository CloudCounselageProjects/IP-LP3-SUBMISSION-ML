# -*- coding: utf-8 -*-
"""
@author: MAYANK
"""

from PIL import Image
import numpy as np
import tensorflow as tf

pred = {0 : 'Red', 1 : 'Green' , 2 : 'Blue'}
model = tf.keras.models.load_model('train_model.h5')

def predict(test_image, i):
    #Convert images to float and resize images
    img = tf.io.read_file(test_image)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (224, 224))
    img = tf.reshape(img, (1, 224, 224, 3))
    #use model to predict
    prediction = model.predict(img, batch_size=1,steps=1)
    predict = np.argmax(prediction)
    print(f'Image {i}  Predict value :' + pred[predict])


for i in range(20):
    predict('testing/' + str(i) +'.jpg', str(i))
