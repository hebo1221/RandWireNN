"""
Example configuration with TensorBoard experiment tracking.
Install: pip install tensorboard
View: tensorboard --logdir=./output/experiments
"""
from RandWireNN_config import cfg

# Enable TensorBoard logging
cfg.USE_TENSORBOARD = True
cfg.USE_WANDB = False

# Experiment settings
cfg.EXPERIMENT_NAME = "rwnn_tensorboard_demo"

# Use modern optimizer
cfg.OPTIMIZER = "adamw"
cfg.LEARNING_RATE = 0.001
cfg.WEIGHT_DECAY = 0.05

# Cosine warmup schedule
cfg.SCHEDULER = "cosine_warmup"
cfg.T_0 = 10

# Mixed precision for speed
cfg.USE_AMP = True

# Model checkpointing
cfg.SAVE_BEST_MODEL = True
cfg.SAVE_CHECKPOINT_FREQ = 10

print("Configuration: TensorBoard Experiment Tracking")
print(f"Experiment name: {cfg.EXPERIMENT_NAME}")
print(f"TensorBoard: {cfg.USE_TENSORBOARD}")
print(f"Optimizer: {cfg.OPTIMIZER}, Scheduler: {cfg.SCHEDULER}")
print("\nAfter training, view results with:")
print("  tensorboard --logdir=./output/experiments")
