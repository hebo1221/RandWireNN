from easydict import EasyDict as edict
import os
__C = edict()
cfg = __C

# Data set config

__C.DATASET_NAME = "ImageNet"
__C.DOWNLOAD = True

# If set to 'True' training will use trained model
__C.LOAD_TRAINED_MODEL = False
__C.TRAINED_MODEL_LOAD_DIR = "./output/model/081_007000.cpt"

#
# Training parameters
#
__C.BATCH_SIZE = 20
__C.EPOCH = 250

#
# Network parameters
#
__C.NN = edict()
__C.NN.REGIME = "SMALL"

# for image color scale
__C.NN.COLOR = 3
# Resizied to 224x224 in torchvision.transforms.Resize()
__C.NN.IMG_SIZE = 224
# Random graph node
__C.NN.NODES = 32
__C.NN.NUM_CLASSES = 1000