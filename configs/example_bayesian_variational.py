"""
Example configuration for Bayesian RandWireNN with Variational Inference.

Uses full Bayesian treatment with weight distributions.
More computationally intensive but provides better uncertainty estimates.

Based on: "Weight Uncertainty in Neural Networks" (Blundell et al., 2015)
         "Bayesian Randomly Wired Neural Network with Variational Inference" (2020)
"""
from RandWireNN_config import cfg

# Enable Bayesian mode with Variational Inference
cfg.USE_BAYESIAN = True
cfg.BAYESIAN_METHOD = "variational"
cfg.VARIATIONAL_PRIOR_STD = 1.0  # Prior standard deviation
cfg.KL_WEIGHT = 1e-5  # Weight for KL divergence loss
cfg.MC_SAMPLES = 30  # Number of samples for uncertainty
cfg.ESTIMATE_UNCERTAINTY = True

# Enable experiment tracking
cfg.USE_TENSORBOARD = True
cfg.EXPERIMENT_NAME = "bayesian_rwnn_variational"

# Optimizer: AdamW works well with variational inference
cfg.OPTIMIZER = "adamw"
cfg.LEARNING_RATE = 0.001  # Often need lower LR for variational
cfg.WEIGHT_DECAY = 0.01  # Lower weight decay since KL handles regularization

# Scheduler
cfg.SCHEDULER = "cosine"

# Training
# cfg.EPOCH = 100  # May need more epochs for convergence

# Model checkpointing
cfg.SAVE_BEST_MODEL = True

print("=" * 60)
print("Bayesian RandWireNN Configuration: Variational Inference")
print("=" * 60)
print(f"Method: {cfg.BAYESIAN_METHOD}")
print(f"Prior std: {cfg.VARIATIONAL_PRIOR_STD}")
print(f"KL weight: {cfg.KL_WEIGHT}")
print(f"MC samples for uncertainty: {cfg.MC_SAMPLES}")
print(f"Uncertainty estimation: {cfg.ESTIMATE_UNCERTAINTY}")
print("\nThis will:")
print("  - Replace layers with Bayesian layers (weight distributions)")
print("  - Add KL divergence to loss function")
print("  - Learn both mean and variance of weights")
print("  - Provide principled uncertainty quantification")
print("  - Generate calibration plots and reliability diagrams")
print("\nNote: Variational inference is more computationally expensive")
print("      but provides better uncertainty estimates.")
print("=" * 60)
