import tensorflow as tf
import pathlib
from tensorflow import keras
import numpy as np

# model path
model = keras.models.load_model(pathlib.Path.cwd()/'trained_model')

# choose photo
sunflower_path = pathlib.Path.cwd() / 'test_img' / '20x b 1.bmp'

img_height = 180
img_width = 180

img = keras.preprocessing.image.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# folder name
class_names = ['bad', 'good']

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)