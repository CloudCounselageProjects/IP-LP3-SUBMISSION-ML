import numpy as np
from keras.preprocessing import image
from keras import Sequential
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split


def model_definition(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(100, 100, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train,
              epochs=90,
              validation_data=(X_test, y_test),
              batch_size=32)

    return model


def pred_color(in_img, model):
    in_img = image.load_img(in_img)
    in_img = image.img_to_array(in_img)
    in_img = in_img / 255
    in_img = np.expand_dims(in_img, axis=0)

    y_hat = model.predict(in_img)

    red = y_hat[0][0]
    green = y_hat[0][1]
    blue = y_hat[0][2]

    return get_color((red, green, blue))


def get_color(value):
    color = None
    red, green, blue = value
    if red > blue and red > green:
        color = 'RED'

    elif blue > red and blue > green:
        color = 'BLUE'

    elif green > blue and green > red:
        color = 'GREEN'

    elif green == blue and (green > red or blue > red):
        color = 'GREEN+BLUE'

    elif green == red and (red > blue or green > blue):
        color = 'GREEN+RED'

    elif blue == red and (red > green or blue > green):
        color = 'BLUE+RED'

    elif blue == red and green == blue:

        if blue == 0:
            color = 'BLACK'
        elif blue == 255:
            color = 'WHITE'
        else:
            color = 'RED+GREEN+BLUE'

    return color
