import numpy as np
import cv2
from keras.preprocessing import image
import tensorflow as tf
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image")
args = vars(ap.parse_args())

labels=["Red","Green","Blue"]
test_image=cv2.imread(args["input"])
image=cv2.resize(test_image,(28,28))
image=image.astype("float32")/255.0
image = np.expand_dims(image, axis=0)
model = tf.keras.models.load_model("rgbcolourclassifiernormalized.h5")
lists = model.predict(image)
print("The color is ",labels[np.argmax(lists)-1])