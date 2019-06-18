from easydict import EasyDict as edict
import os
__C = edict()
cfg = __C

# Data set config

__C.DATASET_NAME = "MNIST"
__C.DOWNLOAD = True

# If set to 'True' training will use trained model
__C.LOAD_TRAINED_MODEL = False
__C.TRAINED_MODEL_LOAD_DIR = "./output/model/MNIST_099_00.cpt"

#
# Training parameters
#
__C.BATCH_SIZE = 128
__C.EPOCH = 250

#
# Network parameters
#
__C.NN = edict()
__C.NN.REGIME = "SMALL"

# for image color scale: gray scale
__C.NN.COLOR = 1
# Actural image is 28x28, but resizied
__C.NN.IMG_SIZE = 32
# Random graph node
__C.NN.NODES = 32
__C.NN.CHANNELS = 78
__C.NN.NUM_CLASSES = 10


