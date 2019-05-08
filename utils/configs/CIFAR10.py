from easydict import EasyDict as edict
import os
__C = edict()
__C.DATA = edict()
cfg = __C


# data set config

__C.DATA.DATASET = "CIFAR10"
# If true, downloads the dataset from the internet and puts it in root directory. 
# If dataset is already downloaded, it is not downloaded again.
__C.DATA.DOWNLOAD = True