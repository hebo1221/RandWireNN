"""
Example configuration for Bayesian RandWireNN with MC Dropout.

Uses Monte Carlo Dropout for uncertainty quantification.
This is the simplest and most efficient method.

Based on: "Dropout as a Bayesian Approximation" (Gal & Ghahramani, 2016)
"""
from RandWireNN_config import cfg

# Enable Bayesian mode with MC Dropout
cfg.USE_BAYESIAN = True
cfg.BAYESIAN_METHOD = "mc_dropout"
cfg.MC_DROPOUT_P = 0.1  # 10% dropout probability
cfg.MC_SAMPLES = 30  # Number of forward passes for uncertainty
cfg.ESTIMATE_UNCERTAINTY = True

# Enable experiment tracking to visualize uncertainty
cfg.USE_TENSORBOARD = True
cfg.EXPERIMENT_NAME = "bayesian_rwnn_mc_dropout"

# Use modern optimizer
cfg.OPTIMIZER = "adamw"
cfg.LEARNING_RATE = 0.001
cfg.WEIGHT_DECAY = 0.05

# Scheduler
cfg.SCHEDULER = "cosine"

# Model checkpointing
cfg.SAVE_BEST_MODEL = True

print("=" * 60)
print("Bayesian RandWireNN Configuration: MC Dropout")
print("=" * 60)
print(f"Method: {cfg.BAYESIAN_METHOD}")
print(f"Dropout probability: {cfg.MC_DROPOUT_P}")
print(f"MC samples for uncertainty: {cfg.MC_SAMPLES}")
print(f"Uncertainty estimation: {cfg.ESTIMATE_UNCERTAINTY}")
print("\nThis will:")
print("  - Add dropout layers throughout the network")
print("  - Enable dropout during inference")
print("  - Perform MC sampling for uncertainty estimation")
print("  - Compute epistemic and total uncertainty")
print("  - Generate calibration plots")
print("=" * 60)
