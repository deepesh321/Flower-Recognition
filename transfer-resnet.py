import sys
import os
import time
import zipfile
import numpy as np
import tensorflow as tf
from PIL import ImageFile
from keras.layers import *
from sklearn.utils import shuffle
from keras.preprocessing import image
from matplotlib import pyplot as plt
from keras.utils import np_utils
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.applications import VGG19, ResNet50, InceptionV3
from keras.applications.vgg16 import preprocess_input
ImageFile.LOAD_TRUNCATED_IMAGES = True


# loading data
img_data_list = []
image1 = 18540
for img in range(image1):
    img_path = 'train' + '/' + str(img) + '.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img_data_list.append(x)

# printing shape of the array
img_data = np.array(img_data_list)
img_data = np.rollaxis(img_data, 1, 0)
img_data = img_data[0]
print(img_data.shape)

num_classes = 102
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')

# loading target value
count = 0
file = open('train.csv', 'r')
for line in file:
    line1 = line.rstrip('\r\n').split(',')
    labels[count] = int(line1[1])-1
    count = count+1
Y = np_utils.to_categorical(labels, num_classes)
file.close()
x, y = shuffle(img_data, Y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# model
model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(2048, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(102, activation='softmax'))
for layer in model.layers[:-6]:
    layer.trainable = False
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
model.summary()

# fitting the model
start = time.time()
model_ResNet50_info = model.fit(
    X_train, y_train, batch_size=32, epochs=10, verbose=2, validation_data=(X_test, y_test))
print('Training time: %s' % (start - time.time()))

# testing
image_list = os.listdir('test')
file = open('test.csv', 'a')
for imag in image_list:
    img_path = 'test' + '/' + imag
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    class1 = np.argmax(preds)
    image1 = imag.split('.')
    file.write(image1[0]+','+str(class1+1))
    file.write('\n')
