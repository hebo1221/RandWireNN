import os
import torch
from easydict import EasyDict as edict
import time

__C = edict()
cfg = __C

# CPU or GPU
__C.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Test mode
__C.TEST_MODE = False

# model config
__C.GRAPH_MODEL = "WS"

# dataset directory
__C.DATASET_DIR = "G:/dataset/"

if not os.path.isdir(__C.DATASET_DIR):
    # default dataset directory
    __C.DATASET_DIR = "./dataset/"
    if not os.path.isdir(__C.DATASET_DIR):
        os.mkdir(__C.DATASET_DIR)

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

# optimizer
__C.LEARNING_RATE = 0.1
__C.MOMENTUM = 0.9
__C.WEIGHT_DECAY = 0.00005
__C.LR_SCHEDULER_STEP = 1

# Debug parameters
__C.PRINT_FREQ = 10
__C.SAVE_FREQ = 1000
__C.MAKE_GRAPH = False
if not os.path.isfile("./output/graph/conv2.yaml"):
    __C.MAKE_GRAPH = True
# For reproducibility
__C.RND_SEED = 3

# Enable Visdom for loss visualization
# install: pip install visdom
# execute: python -m visdom.server
# access:  http://localhost:8097
__C.Visdom = False

if cfg.Visdom:
    from visdom import Visdom
    __C.vis = Visdom()
    __C.loss_window = ""

# Unused


# Enable plotting of generated random graph model
__C.VISUALIZE_GRAPH = False