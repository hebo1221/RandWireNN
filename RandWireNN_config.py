import os
import torch
from easydict import EasyDict as edict
import time

__C = edict()
cfg = __C

# CPU or GPU
__C.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Test mode: skip training
__C.TEST_MODE = False

# set your dataset directory
__C.DATASET_DIR = "C:/dataset/"

# default dataset directory
if not os.path.isdir(__C.DATASET_DIR):
    __C.DATASET_DIR = "./dataset/"
    if not os.path.isdir(__C.DATASET_DIR):
        os.mkdir(__C.DATASET_DIR)

# Graph Model Configuration
# Available models: ER, BA, WS, NWS, PC, RR, COMPLETE, PATH, CYCLE
__C.GRAPH_MODEL = "WS"

# Erdos-Renyi (ER) model parameters
__C.ER_P = 0.2  # Edge probability

# Barabasi-Albert (BA) model parameters
__C.BA_M = 5  # Number of edges to attach from new node

# Watts-Strogatz (WS) model parameters
__C.WS_K = 4  # Each node connected to k nearest neighbors
__C.WS_P = 0.75  # Probability of rewiring

# Newman-Watts-Strogatz (NWS) model parameters
__C.NWS_K = 4  # Each node connected to k nearest neighbors
__C.NWS_P = 0.1  # Probability of adding new edge

# Powerlaw Cluster (PC) model parameters
__C.PC_M = 3  # Number of random edges to add for each new node
__C.PC_P = 0.1  # Probability of adding triangle after edge

# Random Regular (RR) model parameters
__C.RR_D = 4  # Degree of each node (must result in even total degree)

# Optimizer Configuration
__C.OPTIMIZER = "sgd"  # Options: sgd, adam, adamw, lion, rmsprop
__C.LEARNING_RATE = 0.1
__C.MOMENTUM = 0.9
__C.WEIGHT_DECAY = 5e-5
__C.NESTEROV = False  # Use Nesterov momentum for SGD
__C.BETAS = (0.9, 0.999)  # Beta parameters for Adam/AdamW
__C.ADAM_EPS = 1e-8  # Epsilon for Adam/AdamW

# Learning Rate Scheduler Configuration
__C.SCHEDULER = "cosine"  # Options: cosine, cosine_warmup, onecycle, step, multistep, exponential, reduce_on_plateau, none
__C.ETA_MIN = 0  # Minimum learning rate for cosine schedules
__C.T_0 = 10  # Initial restart period for cosine_warmup
__C.T_MULT = 2  # Period multiplier for cosine_warmup
__C.STEP_SIZE = 30  # Step size for step scheduler
__C.MILESTONES = [30, 60, 90]  # Milestones for multistep scheduler
__C.GAMMA = 0.1  # Learning rate decay factor
__C.PCT_START = 0.3  # Percentage of cycle for OneCycleLR warmup
__C.ANNEAL_STRATEGY = 'cos'  # Annealing strategy for OneCycleLR

# Mixed Precision Training
__C.USE_AMP = False  # Automatic Mixed Precision (FP16)
__C.AMP_OPT_LEVEL = "O1"  # AMP optimization level (O0, O1, O2, O3)

# Experiment Tracking
__C.EXPERIMENT_NAME = f"rwnn_{time.strftime('%Y%m%d_%H%M%S')}"  # Experiment name
__C.OUTPUT_DIR = "./output"  # Output directory for experiments
__C.USE_TENSORBOARD = False  # Enable TensorBoard logging
__C.USE_WANDB = False  # Enable Weights & Biases logging
__C.WANDB_PROJECT = "randwirenn"  # W&B project name
__C.WANDB_ENTITY = None  # W&B entity/team name (None for personal)

# Model Checkpointing
__C.SAVE_BEST_MODEL = True  # Save best model based on validation metric
__C.BEST_MODEL_METRIC = "val_acc"  # Metric to track for best model
__C.BEST_MODEL_MODE = "max"  # 'max' for accuracy, 'min' for loss
__C.SAVE_CHECKPOINT_FREQ = 10  # Save checkpoint every N epochs (0 to disable)

# Bayesian Neural Network / Uncertainty Quantification
__C.USE_BAYESIAN = False  # Enable Bayesian layers for uncertainty estimation
__C.BAYESIAN_METHOD = "mc_dropout"  # Options: mc_dropout, variational
__C.MC_DROPOUT_P = 0.1  # Dropout probability for MC Dropout
__C.MC_SAMPLES = 30  # Number of Monte Carlo samples for uncertainty
__C.VARIATIONAL_PRIOR_STD = 1.0  # Prior std for variational layers
__C.KL_WEIGHT = 1e-5  # Weight for KL divergence loss (variational only)
__C.ESTIMATE_UNCERTAINTY = False  # Compute uncertainty during validation

# Debug parameters
__C.PRINT_FREQ = 10
__C.SAVE_FREQ = 1000
__C.VAL_FREQ = 3
__C.MAKE_GRAPH = False
if not os.path.isfile("./output/graph/conv2.yaml"):
    __C.MAKE_GRAPH = True
# For reproducibility
__C.RND_SEED = 3

# Enable Visdom for loss visualization
# install: pip install visdom
# execute: python -m visdom.server
# access:  http://localhost:8097
__C.VISDOM = False

if cfg.VISDOM:
    from visdom import Visdom
    __C.vis = Visdom()
    __C.loss_window = ""


# Unused

# Enable plotting of generated random graph model
__C.VISUALIZE_GRAPH = False