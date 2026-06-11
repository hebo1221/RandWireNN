#!/usr/bin/env python
"""
Quick training demonstration with synthetic data
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils.network import Net
from utils.optimizers import get_optimizer, get_scheduler
from utils.experiment_tracker import ExperimentTracker
import time
import logging
from easydict import EasyDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_synthetic_data(num_samples=1000, img_size=128, num_classes=10):
    """Create synthetic dataset for demonstration"""
    X = torch.randn(num_samples, 3, img_size, img_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)

def create_config():
    """Create configuration"""
    from RandWireNN_config import cfg
    from utils.configs.cifar10_config import cfg as dataset_cfg
    from utils.config_helpers import merge_configs

    # Merge base configs
    merged_cfg = merge_configs([cfg, dataset_cfg])

    # Override with demo settings
    merged_cfg.EPOCH = 3
    merged_cfg.BATCH_SIZE = 32
    merged_cfg.LEARNING_RATE = 0.001
    merged_cfg.WEIGHT_DECAY = 0.01
    merged_cfg.OPTIMIZER = "adamw"
    merged_cfg.SCHEDULER = "step"
    merged_cfg.STEP_SIZE = 1
    merged_cfg.GAMMA = 0.9
    merged_cfg.DEVICE = torch.device("cpu")
    merged_cfg.GRAPH_MODEL = "WS"
    merged_cfg.MAKE_GRAPH = True
    merged_cfg.USE_TENSORBOARD = True
    merged_cfg.EXPERIMENT_NAME = "synthetic_demo"
    merged_cfg.PRINT_FREQ = 10
    merged_cfg.OUTPUT_DIR = "./output"

    return merged_cfg

def train_epoch(model, dataloader, criterion, optimizer, epoch, cfg, scheduler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(cfg.DEVICE), targets.to(cfg.DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % cfg.PRINT_FREQ == 0:
            logger.info(f"  Batch [{batch_idx}/{len(dataloader)}] "
                       f"Loss: {loss.item():.4f} "
                       f"Acc: {100.*correct/total:.2f}%")

    avg_loss = running_loss / len(dataloader)
    acc = 100. * correct / total

    return avg_loss, acc

def validate(model, dataloader, criterion, cfg):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(cfg.DEVICE), targets.to(cfg.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(dataloader)
    acc = 100. * correct / total

    return avg_loss, acc

if __name__ == '__main__':
    print("=" * 70)
    print("RANDWIRENN TRAINING DEMONSTRATION (Synthetic Data)")
    print("=" * 70)

    # Setup configuration
    cfg = create_config()
    logger.info(f"Device: {cfg.DEVICE}")
    logger.info(f"Experiment: {cfg.EXPERIMENT_NAME}")
    logger.info(f"Graph model: {cfg.GRAPH_MODEL}")

    # Create synthetic dataset
    logger.info("Creating synthetic dataset...")
    train_dataset = create_synthetic_data(num_samples=320, img_size=128)
    val_dataset = create_synthetic_data(num_samples=128, img_size=128)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Build model
    logger.info("Building RandWireNN model...")

    # Prepare the graph
    from RandWireNN_train import prepare
    prepare(cfg)

    model = Net(cfg)
    model.to(cfg.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")

    criterion = nn.CrossEntropyLoss()

    # Setup optimizer
    optimizer = get_optimizer(cfg.OPTIMIZER, model.parameters(), cfg)
    logger.info(f"Optimizer: {cfg.OPTIMIZER.upper()} (LR={cfg.LEARNING_RATE})")

    # Setup scheduler
    cfg.STEPS_PER_EPOCH = len(train_loader)
    scheduler = get_scheduler(cfg.SCHEDULER, optimizer, cfg)
    logger.info(f"Scheduler: {cfg.SCHEDULER}")

    # Setup experiment tracking
    tracker = ExperimentTracker(cfg) if cfg.USE_TENSORBOARD else None
    if tracker:
        log_dir = f"{cfg.OUTPUT_DIR}/experiments/{cfg.EXPERIMENT_NAME}/tensorboard"
        logger.info(f"TensorBoard logs: {log_dir}")

    # Training loop
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)

    start_time = time.time()
    best_acc = 0.0

    for epoch in range(cfg.EPOCH):
        logger.info(f"\nEpoch [{epoch+1}/{cfg.EPOCH}]")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, epoch, cfg, scheduler)
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

        # Log metrics
        if tracker:
            tracker.log_metrics({
                'loss': train_loss,
                'acc': train_acc
            }, epoch, prefix='train/')

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, cfg)
        logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        if tracker:
            tracker.log_metrics({
                'loss': val_loss,
                'acc': val_acc
            }, epoch, prefix='val/')

        # Track best model
        if val_acc > best_acc:
            best_acc = val_acc
            logger.info(f"✓ New best model! Acc: {val_acc:.2f}%")

        # Step scheduler
        if scheduler and cfg.SCHEDULER.lower() == 'step':
            scheduler.step()
            logger.info(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Training complete
    elapsed_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Best validation accuracy: {best_acc:.2f}%")

    if tracker:
        log_dir = f"{cfg.OUTPUT_DIR}/experiments/{cfg.EXPERIMENT_NAME}/tensorboard"
        logger.info(f"TensorBoard logs: {log_dir}")

    logger.info("=" * 70)
    logger.info("\nTo view TensorBoard logs:")
    log_dir = f"{cfg.OUTPUT_DIR}/experiments/{cfg.EXPERIMENT_NAME}/tensorboard" if tracker else 'output/experiments'
    logger.info(f"  tensorboard --logdir {log_dir}")
