from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image

img_width,img_height=64,64
train_dataset_dir='dataset/train'
validation_dataset_dir='dataset/validation'
nb_train_samples=800
nb_validation_samples=100
epochs=50
batch_size=20

train_datagen=ImageDataGenerator(rescale=1. /255)
test_datagen=ImageDataGenerator(rescale=1. /255)
train_generator=train_datagen.flow_from_directory(
	r'C:/Users/varsh/Desktop/color classifier/dataset/train',
	target_size=(img_width,img_height),
	batch_size=batch_size,
	class_mode='categorical')
validation_generator=test_datagen.flow_from_directory(
	r'C:/Users/varsh/Desktop/color classifier/dataset/validation',
	target_size=(img_width,img_height),
	batch_size=batch_size,
	class_mode='categorical')

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.summary()

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))
model.summary()

model.compile(
	loss='categorical_crossentropy',
	optimizer='adam',
	metrics=['accuracy'])

history=model.fit_generator(
	train_generator,
	steps_per_epoch=nb_train_samples / batch_size,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=nb_validation_samples/ batch_size)
model.save("ColorClassifier.h5")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
print('accuracy:',sum(acc)/len(acc)*100,'%')
print('val_accuracy:',sum(val_acc)/len(val_acc)*100,'%')



