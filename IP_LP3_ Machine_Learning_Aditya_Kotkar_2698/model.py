#%% Impoting depenedencies
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from IPython.display import display
import os
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#%% Data Preprocessing

#%% Loading Datset

train_dir = './RGB_Dataset/train'
validation_dir = './RGB_Dataset/validation'

#%% Loading R,G,B class files

train_red_dir = os.path.join(train_dir, 'RED')  # directory with our training red pictures
train_green_dir = os.path.join(train_dir, 'GREEN')  # directory with our training green pictures
train_blue_dir = os.path.join(train_dir, 'BLUE')  # directory with our training blue pictures
validation_red_dir = os.path.join(validation_dir, 'RED')  # directory with our validation RED pictures
validation_green_dir = os.path.join(validation_dir, 'GREEN')  # directory with our validation GREEN pictures
validation_blue_dir = os.path.join(validation_dir, 'BLUE')  # directory with our validation BLUE pictures

#%% Calculating images
num_red_tr = len(os.listdir(train_red_dir))
num_green_tr = len(os.listdir(train_green_dir))
num_blue_tr = len(os.listdir(train_blue_dir))

num_red_val = len(os.listdir(validation_red_dir))
num_green_val = len(os.listdir(validation_green_dir))
num_blue_val = len(os.listdir(validation_blue_dir))

total_train = num_red_tr + num_green_tr + num_blue_tr
total_val = num_red_val + num_green_val + num_blue_val

print(total_train + total_val) 

#%% Defining common attributes

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

#%% Data Preparation/ Normalization

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

#%% Prparing training set

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

#%% Preparing validation set

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

#%% Visualizing Training images

sample_training_images, _ = next(train_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
plotImages(sample_training_images[:5])

#%% Model

#%% Defining Model and adding convolution, maxpooling, dropout, dense layers (Hidden layers)

model = Sequential([
    # Hidden
    Conv2D(32, 3, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    Conv2D(32, 3, 3, activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.5),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

#%% Compiling the model

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#%% Summary of model

model.summary()

#%% Training the model

history = model.fit_generator(
    train_data_gen,
    # batch_size=32,
    #steps_per_epoch=total_train,
    epochs=epochs,
    #validation_data=val_data_gen,
    validation_steps=total_val
)

#%% Accuracy of model

valid_loss, valid_accuracy = model.evaluate_generator(val_data_gen)
print("Accuracy after transfer learning: {}".format(valid_accuracy))

#%% Predicting results for test images

'''
    Images:
        validation:
            RED: RED/red_76462.png, RED/red_83633.png, RED/red_94955.png
            GREEN: GREEN/green_22548.png, GREEN/green_42390.png, GREEN/green_66080.png 
            BLUE: BLUE/blue_10133.png, BLUE/blue_30237.png, BLUE/blue_52610.png 
        0 - BLUE, 1 - GREEN, 2 - RED
'''

img =  tf.keras.preprocessing.image.load_img('./RGB_Dataset/validation/image_url', target_size=(IMG_HEIGHT, IMG_WIDTH))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
prediction = model.predict_classes(x)
if prediction==[0] : print("BLUE")
elif prediction == [1] : print("GREEN")
elif prediction == [2] : print("RED")

plt.imshow(img)

#%% Predicting result for custom images

'''
    Images:
        test_28536,png, test__965800.png, test_47318.png, test_22803.png, test_88746.png
        0 - BLUE, 1 - GREEN, 2 - RED
'''

img =  tf.keras.preprocessing.image.load_img('./custom_images/image_url', target_size=(IMG_HEIGHT, IMG_WIDTH))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
prediction = model.predict_classes(x)
if prediction==[0] : print("BLUE")
elif prediction == [1] : print("GREEN")
elif prediction == [2] : print("RED")

plt.imshow(img)





