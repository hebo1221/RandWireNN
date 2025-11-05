"""
Example configuration using SGD with OneCycleLR scheduler.
Great for fast convergence with super-convergence.
"""
from RandWireNN_config import cfg

# SGD with Nesterov momentum
cfg.OPTIMIZER = "sgd"
cfg.LEARNING_RATE = 0.1
cfg.MOMENTUM = 0.9
cfg.WEIGHT_DECAY = 5e-5
cfg.NESTEROV = True  # Use Nesterov momentum

# OneCycleLR for super-convergence
cfg.SCHEDULER = "onecycle"
cfg.PCT_START = 0.3  # 30% of training for warmup
cfg.ANNEAL_STRATEGY = 'cos'  # Cosine annealing

# Mixed precision
cfg.USE_AMP = True

print("Configuration: SGD-Nesterov + OneCycleLR + AMP")
print(f"Optimizer: {cfg.OPTIMIZER}, LR: {cfg.LEARNING_RATE}, Nesterov: {cfg.NESTEROV}")
print(f"Scheduler: {cfg.SCHEDULER}, PCT_START: {cfg.PCT_START}")
print(f"Mixed Precision: {cfg.USE_AMP}")
