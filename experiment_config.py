"""
Quick experiment configuration for RandWireNN demonstration
"""
from RandWireNN_config import cfg

# Quick training for demonstration (5 epochs instead of 250)
cfg.EPOCH = 5
cfg.BATCH_SIZE = 128
cfg.VAL_FREQ = 1  # Validate every epoch

# Modern optimizer and scheduler
cfg.OPTIMIZER = "adamw"
cfg.LEARNING_RATE = 0.001
cfg.WEIGHT_DECAY = 0.01
cfg.SCHEDULER = "cosine"

# Enable mixed precision for faster training
cfg.USE_AMP = True

# Enable experiment tracking
cfg.USE_TENSORBOARD = True
cfg.SAVE_BEST_MODEL = True
cfg.SAVE_CHECKPOINT_FREQ = 2
cfg.EXPERIMENT_NAME = "quick_demo"

# Graph configuration
cfg.GRAPH_MODEL = "WS"  # Watts-Strogatz
cfg.MAKE_GRAPH = True

# Disable Bayesian for quick demo
cfg.USE_BAYESIAN = False
cfg.ESTIMATE_UNCERTAINTY = False

print(f"Experiment configured: {cfg.EXPERIMENT_NAME}")
print(f"Training: {cfg.EPOCH} epochs with {cfg.OPTIMIZER} optimizer")
print(f"Batch size: {cfg.BATCH_SIZE}, LR: {cfg.LEARNING_RATE}")
print(f"Graph model: {cfg.GRAPH_MODEL}")
