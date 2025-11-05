"""
Example configuration using modern AdamW optimizer with Cosine Warmup scheduler.
Recommended for most use cases.
"""
from RandWireNN_config import cfg

# Override optimizer settings
cfg.OPTIMIZER = "adamw"
cfg.LEARNING_RATE = 0.001  # Lower LR for Adam-based optimizers
cfg.WEIGHT_DECAY = 0.05  # Stronger weight decay for AdamW
cfg.BETAS = (0.9, 0.999)

# Use Cosine Annealing with Warm Restarts
cfg.SCHEDULER = "cosine_warmup"
cfg.T_0 = 10  # Restart every 10 epochs
cfg.T_MULT = 2  # Double the period after each restart
cfg.ETA_MIN = 1e-6  # Minimum learning rate

# Enable mixed precision for faster training
cfg.USE_AMP = True

print("Configuration: AdamW + CosineWarmup + AMP")
print(f"Optimizer: {cfg.OPTIMIZER}, LR: {cfg.LEARNING_RATE}")
print(f"Scheduler: {cfg.SCHEDULER}, T_0: {cfg.T_0}")
print(f"Mixed Precision: {cfg.USE_AMP}")
