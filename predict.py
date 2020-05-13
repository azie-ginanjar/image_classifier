import argparse
import json

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

# parse data from arguments input
from utils.predict_utils import predict_classes

parser = argparse.ArgumentParser()

parser.add_argument('path', help='path to image file', action = "store")
parser.add_argument('saved_model', help='path to keras model', action = "store")
parser.add_argument('--top_k', help='number of top predictions', action="store", dest="top_k", type=int, default=5)
parser.add_argument('--category_names', help='path to json file to resolve label name from id', action="store",
                    dest="category_names")

arguments = parser.parse_args()
top_k = arguments.top_k
path = arguments.path
saved_model = arguments.saved_model
category_filename = arguments.category_names

# load keras model
model = tf.keras.models.load_model(saved_model, custom_objects={'KerasLayer':hub.KerasLayer})

# made predict based on our model
image = np.asarray(Image.open(path)).squeeze()
probabilities, classes = predict_classes(path, model, top_k)

# handle class name mapping if json file supplied
if category_filename is not None:
    with open(category_filename, 'r') as f:
        class_names = json.load(f)
    keys = [str(x+1) for x in list(classes)]
    classes = [class_names.get(key) for key in keys]

print('Top {} classes:'.format(top_k))
for i in np.arange(top_k):
    print('----------------------------------------------\n')
    print('Class Name: {}'.format(classes[i]))
    print('Probability: {:.2%}'.format(probabilities[i]))
    print('----------------------------------------------\n')
