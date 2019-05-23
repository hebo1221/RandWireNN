import os
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# optimizer
__C.LEARNING_RATE = 0.005
__C.MOMENTUM = 0.9
__C.WEIGHT_DECAY = 0.00005




# Enable tensorbordx
__C.TENSORBOARDX = False

__C.USE_CUDA = True

# Use CNN instead of RandWireNN
__C.USE_SIMPLE_CNN = False

__C.USE_MULTI_GPU = False

__C.OUTPUT_DIR = "./output/"


__C.PRINT_FREQ = 10


__C.TRAIN = edict()

__C.MAKE_GRAPH_MODE = True

# set to 'True' to run only a single epoch
__C.FAST_MODE = False


__C.TRAIN.EPOCHS = 100
__C.TRAIN.BATCH_SIZE = 128
__C.TRAIN.LR_SCHEDULER = "cosine_lr"
__C.TRAIN.WEIGHT_DECAY = 5e-5


#
# network parameters
#

__C.NN = edict()


# Learning parameters
__C.NN.L2_REG_WEIGHT = 0.0005
__C.NN.MOMENTUM_PER_MB = 0.9


#
# Testing parameters
#

__C.TEST = edict()

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000
## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16




# Debug parameters
__C.NN.DEBUG_OUTPUT = False
__C.NN.GRAPH_TYPE = "png" # "png" or "pdf"

# The learning rate multiplier for all bias weights
__C.NN.BIAS_LR_MULT = 2.0




__C.DRAW_NEGATIVE_ROIS = False
__C.DRAW_UNREGRESSED_ROIS = False
# only for plotting results: boxes with a score lower than this threshold will be considered background
__C.RESULTS_BGR_PLOT_THRESHOLD = 0.1


#
# MISC
#

# For reproducibility
__C.RND_SEED = 3

# Default GPU device id
__C.GPU_ID = 0
