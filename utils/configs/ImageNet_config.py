# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
import os
__C = edict()
__C.DATA = edict()
cfg = __C


# data set config

__C.DATA.DATASET = "ImageNet"
__C.DATA.TRAIN_ROOT = "G:/dataset/ILSVRC2012_img_train/"
__C.DATA.VAL_ROOT = "D:/dataset/ILSVRC2012/ILSVRC2012_img_test/"
__C.DATA.BATCH_SIZE = 100

# For this data set use the following lr factor :
__C.NN.LR_FACTOR = 10.0

