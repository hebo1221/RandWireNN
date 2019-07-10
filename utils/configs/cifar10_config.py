from easydict import EasyDict as edict
import os
__C = edict()
cfg = __C

# Data set config

__C.DATASET_NAME = "CIFAR10"
__C.DOWNLOAD = True

# If set to 'True' training will use trained model
__C.LOAD_TRAINED_MODEL = False
__C.TRAINED_MODEL_LOAD_DIR = "./output/model/CIFAR10_060_00.cpt"

#
# Training parameters
#
__C.BATCH_SIZE = 64
__C.EPOCH = 250

#
# Network parameters
#
__C.NN = edict()
__C.NN.REGIME = "REGULAR"

# for image color scale
__C.NN.COLOR = 3
# 32x32 images resizied to 128x128
__C.NN.IMG_SIZE = 128
# cifar10 32x32 Acc: 85%
# cifar10 64x64 Acc: 90%
# cifar10 128x128 Acc: 91.5%
# Random graph node
__C.NN.NODES = 32
__C.NN.NUM_CLASSES = 10