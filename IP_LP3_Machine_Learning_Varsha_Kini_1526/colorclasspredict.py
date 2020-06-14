from keras.models import load_model

import numpy as np

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

label = ['BLUE','GREEN','RED']

model = load_model('ColorClassifier.h5')
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

classes = model.predict_classes(image)
print(classes)
for i in classes:
	print("The color is ",label[i]) 

   