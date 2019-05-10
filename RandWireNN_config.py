import os
from easydict import EasyDict as edict

__C = edict()
cfg = __C

#
# Training parameters
#

__C.TRAIN = edict()

__C.TRAIN.EPOCHS = 100
__C.TRAIN.BATCH_SIZE = 128
__C.TRAIN.LR_SCHEDULER = "cosine_lr"
__C.TRAIN.WEIGHT_DECAY = 5e-5


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


#
# network parameters
#

__C.NN = edict()


# If set to 'True' training will be skipped if a trained model exists already
__C.NN.MAKE_MODE = True
# set to 'True' to run only a single epoch
__C.NN.FAST_MODE = False
# Debug parameters
__C.NN.DEBUG_OUTPUT = False
__C.NN.GRAPH_TYPE = "png" # "png" or "pdf"

# The learning rate multiplier for all bias weights
__C.NN.BIAS_LR_MULT = 2.0




# Enable plotting of results generally / also plot background boxes / also plot unregressed boxes
__C.VISUALIZE_RESULTS = False
__C.DRAW_NEGATIVE_ROIS = False
__C.DRAW_UNREGRESSED_ROIS = False
# only for plotting results: boxes with a score lower than this threshold will be considered background
__C.RESULTS_BGR_PLOT_THRESHOLD = 0.1


#
# MISC
#

# For reproducibility
__C.RND_SEED = 3

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = False

# Default GPU device id
__C.GPU_ID = 0
