import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

trainPath = '../Data/train'
testPath = '../Data/test'

data = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['blue', 'red', 'orange', 'green', 'violet',
               'indigo', 'yellow']

def getDataset(n, path):
    images = []
    labels = []

    labelsFile = open(path + "/labels.txt", "r")

    for i in range(n):
        images.append(np.asarray(Image.open(path + '/im' + str(i))))
        line = str(labelsFile.readline())
        labels.append(class_names.index(line[:-1]))

    return (np.array(images), np.array(labels))

(test_images, test_labels) = getDataset(50, testPath)
(train_images, train_labels) = getDataset(500, trainPath)

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28, 3)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(7, activation="softmax")
	])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)

model.save("model.h5")
