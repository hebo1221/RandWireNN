"""
Quick test configuration for demonstration.
Small network, few epochs, modern features enabled.
"""
from RandWireNN_config import cfg

# Small model for quick testing
cfg.NODES = 16  # Smaller graph
cfg.CHANNELS = 32  # Fewer channels
cfg.GRAPH_MODEL = "ER"  # Simple Erdos-Renyi
cfg.ER_P = 0.4

# Modern optimizer and scheduler
cfg.OPTIMIZER = "adamw"
cfg.LEARNING_RATE = 0.001
cfg.WEIGHT_DECAY = 0.01
cfg.SCHEDULER = "cosine"

# Quick training settings
cfg.EPOCHS = 2  # Just 2 epochs for demo
cfg.BATCH_SIZE = 32
cfg.USE_AMP = True

# Experiment tracking
cfg.USE_TENSORBOARD = True
cfg.SAVE_BEST_MODEL = True
cfg.EXPERIMENT_NAME = "quick_test_demo"

# Dataset
cfg.DATASET = "CIFAR10"

print("=" * 60)
print("QUICK TEST CONFIGURATION")
print("=" * 60)
print(f"Model: {cfg.NODES} nodes, {cfg.CHANNELS} channels, {cfg.GRAPH_MODEL} graph")
print(f"Optimizer: {cfg.OPTIMIZER} (LR={cfg.LEARNING_RATE}, WD={cfg.WEIGHT_DECAY})")
print(f"Scheduler: {cfg.SCHEDULER}")
print(f"Training: {cfg.EPOCHS} epochs, batch_size={cfg.BATCH_SIZE}")
print(f"Features: AMP={cfg.USE_AMP}, TensorBoard={cfg.USE_TENSORBOARD}")
print("=" * 60)
