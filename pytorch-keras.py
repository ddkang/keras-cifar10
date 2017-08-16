from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from torch.autograd import Variable

import resnet_builder

# Set up loading the test data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


# Load checkpointed model
checkpoint = torch.load('rn20-cifar10.t7')
net = checkpoint['model']
net.eval()
state_dict = net.state_dict()
state_keys_list = list(net.state_dict().keys())
print(state_keys_list)
print(net)


# Make and load weights into Keras model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

keras_model = resnet_builder.PreActResNet20((32, 32, 3), num_classes=10)
# Get list of names, as named in the Keras model
def get_names():
    names = []
    names.append('conv1')

    for i in range(1, 4):
        for j in range(3):
            conv_name_base = 'conv' + str(i) + chr(ord('a')+j) + '_branch'
            bn_name_base = 'bn' + str(i) + chr(ord('a')+j) + '_branch'
            names.append(bn_name_base + '2a')
            names.append(conv_name_base + '2a')
            names.append(bn_name_base + '2b')
            names.append(conv_name_base + '2b')

            if i != 1 and j == 0:
                names.append(conv_name_base + '1')

    names.append('bn1')

    names.append('dense')
    return names

# Get list of parameters
def get_params():
    params = []
    cur_index = 0
    params.append(get_conv_weights(cur_index))
    cur_index += 1

    for i in range(1, 4):
        for j in range(3):
            params.append(get_bn_params(cur_index))
            cur_index += 4
            params.append(get_conv_weights(cur_index))
            cur_index += 1
            params.append(get_bn_params(cur_index))
            cur_index += 4
            params.append(get_conv_weights(cur_index))
            cur_index += 1

            if i != 1 and j == 0:
                params.append(get_conv_weights(cur_index))
                cur_index += 1

    params.append(get_bn_params(cur_index))
    cur_index += 4

    params.append(get_linear_params(cur_index))

    return params


def get_conv_weights(index):
    w = state_dict[state_keys_list[index]].cpu()
    return [np.transpose(w.numpy(), (2, 3, 1, 0))]
    # return [np.transpose(w.numpy(), (2, 3, 1, 0)), np.zeros(w.numpy().shape[0])]

def get_bn_params(index):
    # state dict order for BN parameters are: weight, bias, running mean, & running variance
    w = state_dict[state_keys_list[index]].cpu().numpy()
    b = state_dict[state_keys_list[index + 1]].cpu().numpy()
    rm = state_dict[state_keys_list[index + 2]].cpu().numpy() 
    rv = state_dict[state_keys_list[index + 3]].cpu().numpy()  
    return [w, b, rm, rv]

def get_linear_params(index):
    w = state_dict[state_keys_list[index]].cpu()
    b = state_dict[state_keys_list[index + 1]].cpu()
    return [np.transpose(w.numpy(), (1, 0)), b.numpy()]

# Create dictionary where key = module name and value = module parameter 
def get_params_dictionary():
    names = get_names()
    params = get_params()
    params_dictionary = {name: param for name, param in zip(names, params)}
    return params_dictionary

params_dict = get_params_dictionary()
for name, params in params_dict.items():
    print('>>>>>>>>>>>>>', name)
    keras_model.get_layer(name=name).set_weights(params)


# Run the prediction
count = 0
outputs = []
outputs2 = []
keras_output = []
for i, data in enumerate(testloader):
    x, y = data
    x, y = x.cuda(), y.cuda() 
    inputs = Variable(x)
    output = net(inputs)
    output = output.cpu().data.numpy()

    keras_x = x.cpu().numpy()
    keras_x = np.transpose(keras_x, (0, 2, 3, 1))
    keras_output.append(keras_model.predict(keras_x))

    outputs.append(output)
    count += 1
    if count == 1:
        break

pyt_o1 = np.vstack(outputs)
keras_o = np.vstack(keras_output)

mse = ((pyt_o1 - keras_o) ** 2).sum()
print('=============================')
print('keras & orig-model mse: ', mse)
print('=============================')
