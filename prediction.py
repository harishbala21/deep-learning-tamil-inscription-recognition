# prediction.py

import os
import cv2
import tensorflow as tf

# Append all the categories we want to read
CATEGORIES = []
files = ['1 - Multipart','2 - Unknown']

DATADIR = r'D:\ancient_tamil_deep_learning_project\Ancient_tamil_script_web_deployment\Labelled Dataset - Fig 51'

for directoryfile in os.listdir(DATADIR):
    if(directoryfile in files):
        continue
    CATEGORIES.append(directoryfile)

print(len(CATEGORIES))


# The function prepare(file) allows us to use an image of any size,
# since it automatically resize it to the image size we defined in the first program.
def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# Loading pre-trained data from local machine
model = tf.keras.models.load_model("CNN.keras")


def predict_character(image_path):

    image = prepare(image_path)

    prediction = model.predict([image])
    prediction = list(prediction[0])

    c = CATEGORIES[prediction.index(max(prediction))]

    return c
