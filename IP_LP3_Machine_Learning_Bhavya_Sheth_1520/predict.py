import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import sys

model = keras.models.load_model("model/model.h5")
class_names = ['blue', 'red', 'orange', 'green', 'violet',
               'indigo', 'yellow']

array = []

# for i in range(1, len(sys.argv)+1):
#     array.append(np.asarray(Image.open(str(sys.argv[i]))))
#     prediction = model.predict(array[i-1])
#     print(class_names[prediction])

prediction = model.predict(np.asarray(Image.open('test1')))
print(class_names[prediction])