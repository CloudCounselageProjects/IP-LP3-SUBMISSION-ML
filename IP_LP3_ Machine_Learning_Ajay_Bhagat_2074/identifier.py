import os
from keras.engine.saving import load_model
import tensorflow as tf
from lib import pred_color, get_color

# setting GPU memory growth for no memory glitches
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def model_load():
    model_path = 'model.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Loading Model")
    else:
        print("Model not found, please train model first")

    return model


def run_check(image, mod):
    col = pred_color(image, mod)
    return col


base = 'input\\'
model = model_load()
images = os.listdir(base)
total = len(images)
accu = 0

for img_name in images:

    p_col = run_check(base + img_name, model)

    val = tuple(int(x) for x in img_name.split("(")[1].split(")")[0].split(','))
    c = get_color(val)

    print('Image:', img_name)
    print('Predicted:', p_col, '\nActual:', c)

    if c == p_col:
        accu += 1

print("Total: " + str(total))
print("Accurately Predicted: " + str(accu))