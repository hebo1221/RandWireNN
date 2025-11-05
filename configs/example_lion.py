"""
Example configuration using Lion optimizer.
State-of-the-art optimizer that often outperforms AdamW with less memory.
Requires: pip install lion-pytorch
"""
from RandWireNN_config import cfg

# Lion optimizer settings
cfg.OPTIMIZER = "lion"
cfg.LEARNING_RATE = 0.0001  # Lion uses much lower learning rates
cfg.WEIGHT_DECAY = 0.1  # Lion uses higher weight decay
cfg.BETAS = (0.9, 0.99)  # Default Lion betas

# Cosine schedule works well with Lion
cfg.SCHEDULER = "cosine"
cfg.ETA_MIN = 1e-6

# Mixed precision
cfg.USE_AMP = True

print("Configuration: Lion + Cosine + AMP")
print(f"Optimizer: {cfg.OPTIMIZER}, LR: {cfg.LEARNING_RATE}")
print(f"Weight Decay: {cfg.WEIGHT_DECAY} (Lion uses higher WD)")
print(f"Scheduler: {cfg.SCHEDULER}")
print(f"Mixed Precision: {cfg.USE_AMP}")
print("\nNote: Install Lion with: pip install lion-pytorch")
