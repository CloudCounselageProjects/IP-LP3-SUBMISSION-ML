#### importing the libraries 
import tensorflow as tf
import cv2
import numpy as np

MODEL = tf.keras.models.load_model('trained_model1.h5')        ##### path to the pre trained model
LABELS = {0:'Blue',1:'Green',2:'Red'}                          ##### dictionary for all the labels
path = 'download.png'                                          ##### path to the image to be predicted

def predict(path):                                              ##### Function for the necessary processing
    img = cv2.imread(path)                                      ##### and prediction of the image
    img = cv2.resize(img,(224,224))
    img = img/255.0
    img = np.expand_dims(img,axis=0)
    prediction = MODEL.predict(img)
    return prediction

pred_list = predict(path)                                       ##### list of the probabilities of the image being either of the 3 colors
prediction = np.argmax(pred_list)                               ##### actual image color predicted by the model

print(f'The predicted RGB color the image is closest to is: {LABELS[prediction]} with an accuracy of {pred_list[0][prediction]*100} %')
