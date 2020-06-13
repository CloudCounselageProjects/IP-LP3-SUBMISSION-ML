#Imports
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator


# Define the model
model = Sequential([
    Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'),
    Dropout(rate = 0.1),
    MaxPooling2D(pool_size = (2, 2)),
    Dropout(rate = 0.1),
    Conv2D(32, (3, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2, 2)),
    Dropout(rate = 0.1),
    Flatten(),
    Dense(units=128, activation = 'relu'),
    Dense(units=3, activation = 'softmax')

])

# Compile model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

#Set up parameters
IMG_HEIGHT=64
IMG_WIDTH=64
batch_size=32
epochs=15



#Prevent this step from running during the running of predict.py file
if(__name__=="__main__"):
    
    #Train the model
    train_data_generator = ImageDataGenerator(rescale=1./255)

    test_data_generator = ImageDataGenerator(rescale=1./255)

    training_set = train_data_generator.flow_from_directory('train',
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     batch_size=batch_size,
                                                     class_mode='categorical')

    test_set = test_data_generator.flow_from_directory('test',
                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                batch_size=batch_size,
                                                class_mode='categorical')
    model.fit_generator(training_set,
                             steps_per_epoch=500,
                             epochs=epochs,
                             validation_data=test_set,
                             validation_steps=50)

    model.save('RGB_model.h5')

