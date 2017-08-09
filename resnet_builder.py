import numpy as np

import keras
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization, MaxPooling2D, ZeroPadding2D
from keras.layers import Convolution2D
from keras.models import Model
from keras import layers
from keras.utils import np_utils
from keras import backend as K

if (K.image_data_format() == 'channels_first'):
    bn_axis = 1
else:
    bn_axis = 3

BN_EPS = 1e-5

def my_conv(inp, num_filters, kernel_size_tuple, strides=1, padding='valid', name='name'):
    if strides == 2 and kernel_size_tuple[0] != 1:
        x = keras.layers.convolutional.ZeroPadding2D()(inp)
        x = Convolution2D(num_filters, kernel_size_tuple, strides=(2, 2),
                          use_bias=False, kernel_initializer='he_normal', name=name)(x)
    else:
        if strides == 1:
            x = keras.layers.convolutional.ZeroPadding2D()(inp)
        else:
            x = inp
        x = Conv2D(num_filters, kernel_size_tuple, strides=strides, # padding=padding,
                   use_bias=False, kernel_initializer='he_normal', name=name)(x)
    return x


def PreActBlock(stage, block, inp, numFilters, stride, isConvBlock):
    expansion = 1
    conv_name_base = 'conv' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=BN_EPS, name=bn_name_base + '2a')(inp)
    x = Activation('relu')(x)

    if isConvBlock:
        shortcut = my_conv(x, expansion * numFilters, (1, 1),
                           strides=stride, padding='same',
                           name=conv_name_base + '1')
    else:
        shortcut = inp
    
    x = my_conv(x, numFilters, (3, 3), strides=stride, padding='same', name=conv_name_base + '2a')

    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=BN_EPS, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = my_conv(x, numFilters, (3, 3), padding = 'same', name=conv_name_base + '2b')

    x = layers.add([x, shortcut])
    return x

def make_layer(stage, block, inp, numFilters, numBlocks, stride):
    if stride == 1:
        x = block(stage, 'a', inp, numFilters, stride, False)
    else:
        x = block(stage, 'a', inp, numFilters, stride, True)

    for i in range(numBlocks - 1):
        x = block(stage, chr(ord('b') + i), x, numFilters, 1, False)
    return x

def ResNet_builder(block, num_blocks, input_shape, num_classes, nbf=64):
    img_input = Input(shape=input_shape)
    x = my_conv(img_input, nbf, (3, 3), padding='same', name='conv1')

    for i in xrange(len(num_blocks)):
        x = make_layer(i + 1, block, x, nbf, num_blocks[i], (i != 0) + 1)
        nbf *= 2

    x = BatchNormalization(axis = bn_axis, momentum=0.1, epsilon=BN_EPS, name='bn1')(x)
    x = Activation('relu')(x)

    block_shape = K.int_shape(x)
    x = AveragePooling2D((block_shape[1], block_shape[2]))(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name = 'dense')(x)

    return Model(inputs=img_input, outputs=x)

   
def ResNet18(input_shape, num_classes):
    return ResNet_builder(PreActBlock, [2, 2, 2, 2], input_shape, num_classes)

def PreActResNet20(input_shape, num_classes):
    return ResNet_builder(PreActBlock, [3, 3, 3], input_shape, num_classes, nbf=16)
