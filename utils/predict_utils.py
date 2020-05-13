import tensorflow as tf
from PIL import Image
import numpy as np


def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    image = image.numpy()
    return image


def predict_classes(path, model, top_k=5):
    im = Image.open(path)
    image = np.asarray(im)
    processed_image = process_image(image)

    predictions = model.predict(np.expand_dims(processed_image, axis=0))

    probabilities = - np.partition(-predictions[0], top_k)[:top_k]
    classes = np.argpartition(-predictions[0], top_k)[:top_k]
    return probabilities, classes
