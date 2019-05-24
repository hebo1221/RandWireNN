# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
import os
__C = edict()
cfg = __C

# Data set config

__C.DATASET_NAME = "MNIST"
__C.DOWNLOAD = True
__C.TRAIN_ROOT = "G:/dataset/"
__C.VAL_ROOT = "G:/dataset/"

# If set to 'True' training will use trained model
__C.LOAD_TRAINED_MODEL = False
__C.TRAINED_MODEL_LOAD_DIR = "./output/model/MNIST_012_00.cpt"

#
# Training parameters
#
__C.BATCH_SIZE = 8
__C.EPOCH = 1

#
# Network parameters
#
__C.NN = edict()

__C.NN.REGIME = "SMALL"

# for image color scale
__C.NN.COLOR = 1
# Actural image is 28x28, but resizied in torchvision.transforms.Resize()
__C.NN.IMG_SIZE = 32
# Random graph node
__C.NN.NODES = 32
__C.NN.CHANNELS = 78
__C.NN.NUM_CLASSES = 10


