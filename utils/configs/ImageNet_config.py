# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
import os
__C = edict()
cfg = __C

__C.BATCH_SIZE = 110
__C.EPOCH = 1

# data set config

__C.DATASET_NAME = "ImageNet"
__C.DOWNLOAD = True
__C.DATSET_ROOT = "G:/dataset/"
__C.TRAIN_ROOT = "G:/dataset/"

__C.MODEL_LOAD_DIR = "./output/model/081_007000.cpt"

__C.NN = edict()

__C.NN.IMG_SIZE = 256
__C.NN.COLOR = 3
__C.NN.NODES = 32
__C.NN.CHANNELS = 79
__C.NN.NUM_CLASSES = 1000


# If set to 'True' training will be skipped if a trained model exists already
__C.MAKE_MODE = True

__C.NN.REGIME = "SMALL"

# model config
__C.NN.GRAPH_MODEL = "WS"

# Enable plotting of generated random graph model
__C.VISUALIZE_GRAPH = False

# Erdos-Renyi  model
__C.ER = edict()
__C.ER.P = 0.2
# Barabasi-Albert model
__C.BA = edict()
__C.BA.M =  5
# Watts-Strogatz model
__C.WS = edict()
__C.WS.K = 4
__C.WS.P = 0.75