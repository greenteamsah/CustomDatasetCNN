import tensorboard as TensorBoard 
import os
from pathlib import Path
import random
from keras.optimizers import SGD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K  
from os import listdir
from keras.models import Model, load_model
from keras.layers import Input, Flatten,BatchNormalization,UpSampling2D, Activation,Dense, Dropout,merge,Concatenate,core,Cropping2D
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.losses import binary_crossentropy
import warnings
import cv2
warnings.simplefilter('ignore')
import tensorflow as tf
from keras import backend as K
from keras_preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import MaxPooling2D,UpSampling2D,Conv2D,Dense,Input
from numpy import random,vstack
from numpy import dot,sqrt, array,random
from random import sample,shuffle
from tensorflow.keras import backend as K
from PIL import Image

label_list=[]
for i in range(3817):
    if i%36==1:
        label_list.append(i)

def label_file(file):
    word_label = file.split('.')[0]
    return int(word_label)
        
def label_img(img):
    word_label = img.split('.')[0]
    for i in range(len(label_list)):
        if int(word_label) < label_list[i]:
            return (np.abs((int(word_label)-1))//36)

#print(label_img('3781.bmp'))

from sklearn.model_selection import train_test_split

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import re
from tqdm import tqdm

# Pillow
import PIL
from PIL import Image


PATH = os.getcwd()

data_path = PATH+'/SDUMLA_All/'
data_dir_list = os.listdir(data_path)


# Use Pillow library to convert an input jpeg to a 8 bit grey scale image array for processing.
def jpeg_to_8_bit_greyscale(path, maxsize):
        img = Image.open(path).convert('L')   # convert image to 8-bit grayscale
        # Make aspect ratio as 1:1, by applying image crop.
    # Please note, croping works for this data set, but in general one
    # needs to locate the subject and then crop or scale accordingly.
        WIDTH, HEIGHT = img.size
        if WIDTH != HEIGHT:
                m_min_d = min(WIDTH, HEIGHT)
                img = img.crop((0, 0, m_min_d, m_min_d))
        # Scale the image to the requested maxsize by Anti-alias sampling.
        img.thumbnail(maxsize, PIL.Image.ANTIALIAS)
        return np.asarray(img)
    
def load_image_dataset(path_dir, maxsize):
        images = []
        labels = []
        os.chdir(path_dir)
        for file in glob.glob("*.bmp"):
                img = jpeg_to_8_bit_greyscale(file, maxsize)
                print(file)
                print(label_img(file))
                images.append(img)
                labels.append(label_file(file)//36)
                
                    
        return (np.asarray(images), np.asarray(labels))
    
maxsize = 128, 128

(train_images, train_labels) = load_image_dataset(data_path, maxsize)

X_train, X_valid, y_train, y_valid = train_test_split(train_images, train_labels, test_size=0.50, random_state=2019)


class_names = [i+1 for i in range(3816)]
class_names = list(map(str,class_names))

print(train_images.shape)
print(train_labels)

def display_images(images, labels):
        plt.figure(figsize=(10,10))
        grid_size = min(64, len(images))
        for i in range(grid_size):
                plt.subplot(8, 8, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(images[i], cmap=plt.cm.binary)
                plt.xlabel(class_names[labels[i]])
                
display_images(train_images, train_labels)
plt.show()

train_images = train_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(128, 128)),
        keras.layers.Dense(128, activation=tf.nn.sigmoid),
        keras.layers.Dense(16, activation=tf.nn.sigmoid),
    keras.layers.Dense(107, activation=tf.nn.softmax)
])

sgd = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.7, nesterov=True)

model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)

validation_loss, validation_acc = model.evaluate(X_valid, y_valid)
print('Validation accuracy:', validation_acc)

predictions = model.predict(X_valid)

print(predictions)

display_images(X_valid, np.argmax(predictions, axis = 1))
plt.show()