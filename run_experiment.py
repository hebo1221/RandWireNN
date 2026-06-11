#!/usr/bin/env python
"""
Run a quick training experiment with modernized features
"""
import torch
from RandWireNN_train import train, validate, prepare
from utils.network import Net
from utils.config_helpers import merge_configs
from utils.dataloader import train_data_loader, val_data_loader
from utils.optimizers import get_optimizer, get_scheduler
from utils.experiment_tracker import ExperimentTracker, BestModelTracker
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_configuration():
    # Load base configs
    from RandWireNN_config import cfg as network_cfg
    from utils.configs.cifar10_config import cfg as dataset_cfg

    # Apply experiment configuration (this modifies network_cfg)
    import experiment_config

    return merge_configs([network_cfg, dataset_cfg])


if __name__ == '__main__':
    print("=" * 70)
    print("RANDWIRENN TRAINING EXPERIMENT")
    print("=" * 70)

    cfg = get_configuration()

    logger.info(f"Device: {cfg.DEVICE}")
    logger.info(f"Experiment: {cfg.EXPERIMENT_NAME}")
    logger.info(f"Dataset: {cfg.DATASET_NAME}")
    logger.info(f"Epochs: {cfg.EPOCH}, Batch size: {cfg.BATCH_SIZE}")
    logger.info(f"Optimizer: {cfg.OPTIMIZER.upper()}, Scheduler: {cfg.SCHEDULER}")

    # Prepare environment and load data
    prepare(cfg)
    train_loader = train_data_loader(cfg)
    val_loader = val_data_loader(cfg)

    # Build model
    logger.info("Building model...")
    model = Net(cfg)
    model.to(cfg.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")

    criterion = torch.nn.CrossEntropyLoss().to(cfg.DEVICE)

    # Setup optimizer
    optimizer = get_optimizer(cfg.OPTIMIZER, model.parameters(), cfg)
    logger.info(f"Optimizer: {cfg.OPTIMIZER.upper()} (LR={cfg.LEARNING_RATE}, WD={cfg.WEIGHT_DECAY})")

    # Setup scheduler
    cfg.STEPS_PER_EPOCH = len(train_loader)
    scheduler = get_scheduler(cfg.SCHEDULER, optimizer, cfg)
    logger.info(f"Scheduler: {cfg.SCHEDULER}")

    # Setup mixed precision
    scaler = None
    if cfg.USE_AMP and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Mixed precision (AMP) enabled")

    # Setup experiment tracking
    tracker = ExperimentTracker(cfg) if cfg.USE_TENSORBOARD else None
    best_model_tracker = BestModelTracker(cfg.BEST_MODEL_METRIC, cfg.BEST_MODEL_MODE) if cfg.SAVE_BEST_MODEL else None

    if tracker:
        logger.info(f"TensorBoard logging to: {tracker.log_dir}")

    # Training loop
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)

    start_time = time.time()
    best_acc = 0.0

    for epoch in range(cfg.EPOCH):
        logger.info(f"\nEpoch [{epoch+1}/{cfg.EPOCH}]")

        # Train
        train_metrics = train(train_loader, model, criterion, optimizer, epoch, cfg,
                             scaler=scaler, scheduler=scheduler, tracker=tracker)

        # Validate
        if epoch % cfg.VAL_FREQ == 0 or epoch == cfg.EPOCH - 1:
            val_loss, val_acc = validate(val_loader, model, criterion, cfg)

            logger.info(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            # Track metrics
            if tracker:
                tracker.log_metrics({'loss': val_loss, 'acc': val_acc}, epoch, prefix='val/')

            # Save best model
            if best_model_tracker:
                val_metrics = {cfg.BEST_MODEL_METRIC: val_acc}
                improved = best_model_tracker.update(val_metrics, epoch, model, tracker)
                if improved:
                    logger.info(f"✓ New best model! Acc: {val_acc:.2f}%")
                    best_acc = val_acc

        # Save checkpoint
        if cfg.SAVE_CHECKPOINT_FREQ > 0 and (epoch + 1) % cfg.SAVE_CHECKPOINT_FREQ == 0:
            checkpoint_path = f"{cfg.OUTPUT_DIR}/experiments/{cfg.EXPERIMENT_NAME}/checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Training complete
    elapsed_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info(f"Total time: {elapsed_time/60:.2f} minutes")
    logger.info(f"Best validation accuracy: {best_acc:.2f}%")

    if tracker:
        logger.info(f"TensorBoard logs: {tracker.log_dir}")
        tracker.close()

    logger.info("=" * 70)
