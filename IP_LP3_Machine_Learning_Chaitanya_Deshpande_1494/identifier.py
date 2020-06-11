import os
from keras.engine.saving import load_model
import tensorflow as tf
from model_definition import predict_color, get_color


def model_load():
    model_path = 'model.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Loading Model")
    else:
        print("Model not found!")

    return model


def run_check(image, mod):
    col = predict_color(image, mod)
    return col


dir = 'test_images\\'
model = model_load()
images = os.listdir(dir)
total = len(images)
acc = 0

for img_name in images:

    p_col = run_check(dir + img_name, model)

    l = img_name.split("_")
    temp = l[2]
    l[2] = temp.split(".")[0]
    val = tuple(int(i) for i in l)
    c = get_color(val)

    print('Image:', img_name)
    print('Predicted:', p_col, '\nActual:', c)

    if c == p_col:
        acc += 1

print("Total: " + str(total))
print("Accurately Predicted: " + str(acc))
