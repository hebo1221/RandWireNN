"""
Modern optimizer implementations and factory functions.
Includes AdamW, Lion, and other state-of-the-art optimizers.
"""

import torch
import torch.optim as optim
from typing import Any, Iterator
import logging

logger = logging.getLogger(__name__)


def get_optimizer(name: str, parameters: Iterator, cfg: Any) -> optim.Optimizer:
    """
    Factory function to get optimizer by name.

    Args:
        name: Optimizer name ('sgd', 'adam', 'adamw', 'lion')
        parameters: Model parameters
        cfg: Configuration object with optimizer settings

    Returns:
        PyTorch optimizer instance
    """
    name = name.lower()

    if name == 'sgd':
        logger.info(f"Using SGD optimizer with lr={cfg.LEARNING_RATE}, momentum={cfg.MOMENTUM}")
        return optim.SGD(
            parameters,
            lr=cfg.LEARNING_RATE,
            momentum=cfg.MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY,
            nesterov=getattr(cfg, 'NESTEROV', False)
        )

    elif name == 'adam':
        logger.info(f"Using Adam optimizer with lr={cfg.LEARNING_RATE}")
        return optim.Adam(
            parameters,
            lr=cfg.LEARNING_RATE,
            betas=getattr(cfg, 'BETAS', (0.9, 0.999)),
            weight_decay=cfg.WEIGHT_DECAY,
            eps=getattr(cfg, 'ADAM_EPS', 1e-8)
        )

    elif name == 'adamw':
        logger.info(f"Using AdamW optimizer with lr={cfg.LEARNING_RATE}")
        return optim.AdamW(
            parameters,
            lr=cfg.LEARNING_RATE,
            betas=getattr(cfg, 'BETAS', (0.9, 0.999)),
            weight_decay=cfg.WEIGHT_DECAY,
            eps=getattr(cfg, 'ADAM_EPS', 1e-8)
        )

    elif name == 'lion':
        try:
            from lion_pytorch import Lion
            logger.info(f"Using Lion optimizer with lr={cfg.LEARNING_RATE}")
            return Lion(
                parameters,
                lr=cfg.LEARNING_RATE,
                betas=getattr(cfg, 'BETAS', (0.9, 0.99)),
                weight_decay=cfg.WEIGHT_DECAY
            )
        except ImportError:
            logger.warning("Lion optimizer not available. Install with: pip install lion-pytorch")
            logger.warning("Falling back to AdamW")
            return optim.AdamW(
                parameters,
                lr=cfg.LEARNING_RATE,
                betas=getattr(cfg, 'BETAS', (0.9, 0.999)),
                weight_decay=cfg.WEIGHT_DECAY
            )

    elif name == 'rmsprop':
        logger.info(f"Using RMSprop optimizer with lr={cfg.LEARNING_RATE}")
        return optim.RMSprop(
            parameters,
            lr=cfg.LEARNING_RATE,
            alpha=getattr(cfg, 'RMSPROP_ALPHA', 0.99),
            weight_decay=cfg.WEIGHT_DECAY,
            momentum=cfg.MOMENTUM
        )

    else:
        raise ValueError(f"Unknown optimizer: {name}. Choose from: sgd, adam, adamw, lion, rmsprop")


def get_scheduler(name: str, optimizer: optim.Optimizer, cfg: Any) -> Any:
    """
    Factory function to get learning rate scheduler by name.

    Args:
        name: Scheduler name
        optimizer: PyTorch optimizer
        cfg: Configuration object

    Returns:
        PyTorch scheduler instance
    """
    name = name.lower()

    if name == 'cosine':
        logger.info(f"Using CosineAnnealingLR scheduler with T_max={cfg.EPOCH}")
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.EPOCH,
            eta_min=getattr(cfg, 'ETA_MIN', 0)
        )

    elif name == 'cosine_warmup':
        logger.info("Using CosineAnnealingWarmRestarts scheduler")
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=getattr(cfg, 'T_0', 10),
            T_mult=getattr(cfg, 'T_MULT', 2),
            eta_min=getattr(cfg, 'ETA_MIN', 0)
        )

    elif name == 'onecycle':
        logger.info(f"Using OneCycleLR scheduler")
        steps_per_epoch = getattr(cfg, 'STEPS_PER_EPOCH', 100)
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.LEARNING_RATE,
            epochs=cfg.EPOCH,
            steps_per_epoch=steps_per_epoch,
            pct_start=getattr(cfg, 'PCT_START', 0.3),
            anneal_strategy=getattr(cfg, 'ANNEAL_STRATEGY', 'cos')
        )

    elif name == 'step':
        logger.info(f"Using StepLR scheduler")
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(cfg, 'STEP_SIZE', 30),
            gamma=getattr(cfg, 'GAMMA', 0.1)
        )

    elif name == 'multistep':
        logger.info(f"Using MultiStepLR scheduler")
        milestones = getattr(cfg, 'MILESTONES', [30, 60, 90])
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=getattr(cfg, 'GAMMA', 0.1)
        )

    elif name == 'exponential':
        logger.info(f"Using ExponentialLR scheduler")
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=getattr(cfg, 'GAMMA', 0.95)
        )

    elif name == 'reduce_on_plateau':
        logger.info("Using ReduceLROnPlateau scheduler")
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=getattr(cfg, 'FACTOR', 0.1),
            patience=getattr(cfg, 'PATIENCE', 10),
            verbose=True
        )

    elif name == 'none':
        logger.info("No scheduler used")
        return None

    else:
        raise ValueError(
            f"Unknown scheduler: {name}. Choose from: cosine, cosine_warmup, "
            f"onecycle, step, multistep, exponential, reduce_on_plateau, none"
        )
