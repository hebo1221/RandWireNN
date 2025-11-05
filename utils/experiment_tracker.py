"""
Experiment tracking utilities supporting TensorBoard and Weights & Biases.
Provides unified interface for logging metrics, hyperparameters, and model artifacts.
"""

import os
import logging
from typing import Any, Dict, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Unified experiment tracker supporting multiple backends.

    Supports:
    - TensorBoard (built-in with PyTorch)
    - Weights & Biases (optional, requires wandb)
    - Local JSON logging (always available)
    """

    def __init__(self, cfg: Any, experiment_name: Optional[str] = None):
        self.cfg = cfg
        self.experiment_name = experiment_name or cfg.EXPERIMENT_NAME
        self.use_tensorboard = getattr(cfg, 'USE_TENSORBOARD', False)
        self.use_wandb = getattr(cfg, 'USE_WANDB', False)

        self.tb_writer = None
        self.wandb_run = None

        # Setup experiment directory
        self.exp_dir = Path(cfg.OUTPUT_DIR) / "experiments" / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize backends
        self._init_tensorboard()
        self._init_wandb()
        self._init_json_logger()

        logger.info(f"Experiment: {self.experiment_name}")
        logger.info(f"Output dir: {self.exp_dir}")

    def _init_tensorboard(self):
        """Initialize TensorBoard writer."""
        if not self.use_tensorboard:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.exp_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info(f"TensorBoard logging enabled: {tb_dir}")
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.use_tensorboard = False

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if not self.use_wandb:
            return

        try:
            import wandb

            # Get wandb config
            wandb_project = getattr(self.cfg, 'WANDB_PROJECT', 'randwirenn')
            wandb_entity = getattr(self.cfg, 'WANDB_ENTITY', None)

            # Initialize wandb
            self.wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=self.experiment_name,
                config=self._config_to_dict(self.cfg),
                dir=str(self.exp_dir)
            )
            logger.info(f"Weights & Biases logging enabled: {wandb_project}")
        except ImportError:
            logger.warning("Weights & Biases not available. Install with: pip install wandb")
            self.use_wandb = False
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.use_wandb = False

    def _init_json_logger(self):
        """Initialize JSON metrics logging."""
        self.json_log_file = self.exp_dir / "metrics.jsonl"
        self.config_file = self.exp_dir / "config.json"

        # Save configuration
        with open(self.config_file, 'w') as f:
            json.dump(self._config_to_dict(self.cfg), f, indent=2)
        logger.info(f"Config saved: {self.config_file}")

    def _config_to_dict(self, cfg) -> Dict:
        """Convert config object to dictionary."""
        config_dict = {}
        for key in dir(cfg):
            if not key.startswith('_'):
                value = getattr(cfg, key)
                # Skip non-serializable objects
                if isinstance(value, (str, int, float, bool, list, tuple, dict, type(None))):
                    config_dict[key] = value
        return config_dict

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log metrics to all enabled backends.

        Args:
            metrics: Dictionary of metric names to values
            step: Global step/epoch number
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        # TensorBoard
        if self.tb_writer:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(f"{prefix}{name}", value, step)

        # Weights & Biases
        if self.wandb_run:
            import wandb
            wandb_metrics = {f"{prefix}{name}": value for name, value in metrics.items()}
            wandb_metrics['step'] = step
            wandb.log(wandb_metrics)

        # JSON logging
        log_entry = {
            'step': step,
            'prefix': prefix,
            **metrics
        }
        with open(self.json_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters with final metrics."""
        if self.tb_writer:
            self.tb_writer.add_hparams(hparams, metrics)

    def save_model(self, model, name: str = "best_model.pth"):
        """Save model checkpoint."""
        import torch
        save_path = self.exp_dir / name
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved: {save_path}")

        # Log to wandb if enabled
        if self.wandb_run:
            import wandb
            wandb.save(str(save_path))

    def log_graph_structure(self, graph, name: str = "graph"):
        """Log graph structure visualization."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx

            fig, ax = plt.subplots(figsize=(10, 10))
            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, ax=ax, node_color='lightblue',
                   node_size=500, with_labels=True, arrows=True)

            # Save figure
            graph_path = self.exp_dir / f"{name}.png"
            plt.savefig(graph_path, dpi=150, bbox_inches='tight')
            plt.close()

            # Log to backends
            if self.tb_writer:
                from PIL import Image
                import torchvision.transforms as transforms
                img = Image.open(graph_path)
                img_tensor = transforms.ToTensor()(img)
                self.tb_writer.add_image(f"graphs/{name}", img_tensor, 0)

            if self.wandb_run:
                import wandb
                wandb.log({f"graphs/{name}": wandb.Image(str(graph_path))})

            logger.info(f"Graph visualization saved: {graph_path}")
        except Exception as e:
            logger.warning(f"Failed to log graph structure: {e}")

    def finish(self):
        """Close all logging backends."""
        if self.tb_writer:
            self.tb_writer.close()
            logger.info("TensorBoard writer closed")

        if self.wandb_run:
            import wandb
            wandb.finish()
            logger.info("Weights & Biases run finished")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


class BestModelTracker:
    """Track and save best model based on validation metrics."""

    def __init__(self, metric_name: str = 'val_acc', mode: str = 'max'):
        """
        Args:
            metric_name: Metric to track for best model
            mode: 'max' for accuracy, 'min' for loss
        """
        self.metric_name = metric_name
        self.mode = mode
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = 0

    def update(self, metrics: Dict[str, float], epoch: int, model, tracker: ExperimentTracker) -> bool:
        """
        Update best model if current metrics are better.

        Returns:
            True if this is a new best model
        """
        if self.metric_name not in metrics:
            return False

        current_value = metrics[self.metric_name]
        is_better = (
            (self.mode == 'max' and current_value > self.best_value) or
            (self.mode == 'min' and current_value < self.best_value)
        )

        if is_better:
            self.best_value = current_value
            self.best_epoch = epoch
            tracker.save_model(model, "best_model.pth")
            logger.info(f"New best model! {self.metric_name}={current_value:.4f} at epoch {epoch}")
            return True

        return False
