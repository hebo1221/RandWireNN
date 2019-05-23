# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
import os
__C = edict()
cfg = __C

__C.BATCH_SIZE = 8
__C.EPOCH = 1

# data set config

__C.DATASET_NAME = "CIFAR10"
__C.DOWNLOAD = False
__C.DATSET_ROOT = "G:/dataset/"
__C.TRAIN_ROOT = "G:/dataset/"
__C.VAL_ROOT = "D:/dataset/ILSVRC2012/ILSVRC2012_img_test/"

__C.MODEL_LOAD = True
__C.MODEL_LOAD_DIR = "./output/model/074_000000.cpt"

__C.NN = edict()

__C.NN.COLOR = 3
__C.NN.IMG_SIZE = 32
__C.NN.NODES = 32
__C.NN.CHANNELS = 84
__C.NN.NUM_CLASSES = 10


# If set to 'True' training will be skipped if a trained model exists already
__C.MAKE_MODE = False