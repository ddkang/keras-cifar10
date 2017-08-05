import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import six
import os
import time
import csv
import h5py

from collections import OrderedDict
from collections import Iterable
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback
from keras.datasets import cifar10
from keras import backend as K

from resnet_builder import ResNet18

batch_size = 128
num_epochs = 350
num_classes = 10
data_augmentation = True

def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.1
    epoch_drop = 100
    if epoch < 50:
        return initial_lrate
    lrate = initial_lrate * math.pow(drop, math.floor((epoch - 50)/epoch_drop))
    return lrate

# Preprocess train and test set data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

if (K.image_data_format() == 'channels_first'):
    for i in range(x_train.shape[1]):
        mean = np.mean(x_train[:,i,:,:])
        std_dev = np.std(x_train[:,i,:,:])
        x_train[:,i,:,:] -= mean
        x_train[:,i,:,:] /= std_dev
        x_test[:,i,:,:] -= mean
        x_test[:,i,:,:] /= std_dev
else:
    for i in range(x_train.shape[3]):
        mean = np.mean(x_train[:,:,:,i])
        std_dev = np.std(x_train[:,:,:,i])
        x_train[:,:,:,i] -= mean
        x_train[:,:,:,i] /= std_dev
        x_test[:,:,:,i] -= mean
        x_test[:,:,:,i] /= std_dev

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

img_input = Input(shape=x_train.shape[1:])

model = ResNet18(x_train.shape[1:], num_classes=num_classes)


datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode='constant',
            cval=0,
            horizontal_flip=True,
            vertical_flip=False)
datagen.fit(x_train)

sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=1, validation_data=(x_test, y_test))
for num_epochs, lr_rate in [(150, 0.01), (100, 0.001), ]:#(100, 0.001)]:
    sgd = SGD(lr=lr_rate, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    history = model.fit_generator(datagen.flow(x_train, y_train, 
                                               batch_size=batch_size),
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  epochs=num_epochs,
                                  # callbacks=callbacks_list,
                                  verbose=1,
                                  validation_data=(x_test, y_test))


if (K.image_data_format() == 'channels_first'):
    model.save('cifar10-resnet18-pa-th.h5')
else:
    model.save('cifar10-resnet18-pa-tf.h5')

scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
