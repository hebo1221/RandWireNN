import os
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Use CNN instead of RandWireNN
__C.USE_SIMPLE_CNN = False

# Use cuda
__C.USE_CUDA = True

# model config
__C.GRAPH_MODEL = "WS"

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
__C.LEARNING_RATE = 0.005
__C.MOMENTUM = 0.9
__C.WEIGHT_DECAY = 0.00005

# Debug parameters
__C.PRINT_FREQ = 10
__C.SAVE_FREQ = 1000
__C.MAKE_GRAPH = False





# Unused

__C.LR_SCHEDULER = "cosine_lr"
__C.WEIGHT_DECAY = 5e-5

# Enable plotting of generated random graph model
__C.VISUALIZE_GRAPH = False

# Enable tensorbordx
__C.TENSORBOARDX = False
__C.USE_MULTI_GPU = False

# Learning parameters
__C.L2_REG_WEIGHT = 0.0005
__C.MOMENTUM_PER_MB = 0.9
# Debug parameters
__C.DEBUG_OUTPUT = False
__C.GRAPH_TYPE = "png" # "png" or "pdf"

# The learning rate multiplier for all bias weights
__C.BIAS_LR_MULT = 2.0


#
# MISC
#

# For reproducibility
__C.RND_SEED = 3

# Default GPU device id
__C.GPU_ID = 0
