import numpy as np
import os, time
from keras.applications.vgg16 import VGG16, decode_predictions
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten, merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model


# loading the model
model = load_model('model_v1.h5')
# you can test with any other picture 
img_path = 'c12.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)
x = np.array([x])

img_data = np.array(x)
img_data = np.rollaxis(img_data,1,0)
img_data=img_data[0]

preds = model.predict(img_data)
print(preds)
