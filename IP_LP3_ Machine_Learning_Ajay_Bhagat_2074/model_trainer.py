# Importing all necessary libraries
import pandas as pd
import numpy as np
from keras.preprocessing import image
from data_creator import get_dataset
import tensorflow as tf

from lib import model_definition

# setting GPU memory growth for no memory glitches
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

base = 'dataset\\'
total_images = get_dataset(base)

data = list()
for x in total_images:
    ent = [x, x.split('_')[0]]
    data.append(ent)

train = pd.DataFrame(data, columns=['image', 'color'])

train = pd.concat([train.drop('color', axis=1),
                   pd.get_dummies(train['color'],
                                  prefix='color',
                                  prefix_sep='_',
                                  dummy_na=False)],
                  axis=1)


train_image = []
for i in range(len(total_images)):
    img = image.load_img(base + train['image'][i])
    img = image.img_to_array(img)
    img = img / 255
    train_image.append(img)

X = np.array(train_image)
y = np.array(train[['color_red', 'color_green', 'color_blue']])

model_path = 'model.h5'

model = model_definition(X, y)
model.save(model_path)
print('Model saved to:', model_path)
