import torch
from RandWireNN_train import train, validate, prepare
from utils.network import Net
from utils.config_helpers import merge_configs
from utils.dataloader import train_data_loader, val_data_loader
from utils.optimizers import get_optimizer, get_scheduler
from utils.experiment_tracker import ExperimentTracker, BestModelTracker
from utils.bayesian_layers import convert_to_bayesian, compute_kl_loss
from utils.uncertainty import UncertaintyEstimator, analyze_uncertainty
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_configuration():
    # load configs for base network and data set
    from RandWireNN_config import cfg as network_cfg
    from utils.configs.cifar10_config import cfg as dataset_cfg
    # for the MNIST data set use:     from utils.configs.mnist_config import cfg as dataset_cfg
    # for the CIFAR10 data set use:     from utils.configs.cifar10_config import cfg as dataset_cfg
    # for the ImageNet data set use:    from utils.configs.ImageNet_config import cfg as dataset_cfg
    
    return merge_configs([network_cfg, dataset_cfg])


if __name__ == '__main__':
    cfg = get_configuration()
    prepare(cfg)
    train_loader = train_data_loader(cfg)
    val_loader = val_data_loader(cfg)
    model = Net(cfg)

    # Convert to Bayesian if enabled
    if cfg.USE_BAYESIAN:
        use_variational = (cfg.BAYESIAN_METHOD == "variational")
        model = convert_to_bayesian(model, dropout_p=cfg.MC_DROPOUT_P, use_variational=use_variational)
        logger.info(f"Converted to Bayesian model using {cfg.BAYESIAN_METHOD}")
        if use_variational:
            logger.info(f"KL weight: {cfg.KL_WEIGHT}")

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model.to(cfg.DEVICE)

    criterion = torch.nn.CrossEntropyLoss().to(cfg.DEVICE)

    # Store KL weight in cfg for training
    if cfg.USE_BAYESIAN and cfg.BAYESIAN_METHOD == "variational":
        cfg.compute_kl_loss = True
    else:
        cfg.compute_kl_loss = False

    # Use modern optimizer factory
    optimizer = get_optimizer(cfg.OPTIMIZER, model.parameters(), cfg)

    # Calculate steps per epoch for OneCycleLR
    cfg.STEPS_PER_EPOCH = len(train_loader)

    # Use scheduler factory
    scheduler = get_scheduler(cfg.SCHEDULER, optimizer, cfg)

    # Setup mixed precision training
    scaler = None
    if cfg.USE_AMP and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Mixed precision training (AMP) enabled")

    # Initialize experiment tracker
    tracker = ExperimentTracker(cfg) if (cfg.USE_TENSORBOARD or cfg.USE_WANDB) else None
    best_model_tracker = BestModelTracker(cfg.BEST_MODEL_METRIC, cfg.BEST_MODEL_MODE) if cfg.SAVE_BEST_MODEL else None

    # Log graph structure if tracker is enabled
    if tracker and cfg.MAKE_GRAPH:
        from utils.graph import load_graph
        try:
            graph = load_graph('./output/graph/conv3.yaml')
            tracker.log_graph_structure(graph, name='conv3_graph')
        except:
            pass

    if cfg.LOAD_TRAINED_MODEL:
        model.load_state_dict(torch.load(cfg.TRAINED_MODEL_LOAD_DIR))

    if not cfg.TEST_MODE:
        start = time.time()
        for epoch in range(cfg.EPOCH+1):
            # Pass scheduler and tracker to train function
            train_metrics = train(train_loader, model, criterion, optimizer, epoch, cfg,
                                 scaler=scaler, scheduler=scheduler, tracker=tracker)

            # Step scheduler (handle different scheduler types)
            if cfg.SCHEDULER.lower() == 'reduce_on_plateau':
                # ReduceLROnPlateau needs validation loss
                val_loss, acc = validate(val_loader, model, criterion, cfg)
                if scheduler is not None:
                    scheduler.step(val_loss)
            elif cfg.SCHEDULER.lower() != 'onecycle' and scheduler is not None:
                # OneCycleLR steps per batch, not per epoch
                scheduler.step()

            # Validation
            if epoch % cfg.VAL_FREQ == 0:
                val_loss, acc = validate(val_loader, model, criterion, cfg)

                # Log validation metrics
                if tracker:
                    tracker.log_metrics({'loss': val_loss, 'acc': acc}, epoch, prefix='val/')

                # Track best model
                if best_model_tracker:
                    val_metrics = {'val_loss': val_loss, 'val_acc': acc}
                    best_model_tracker.update(val_metrics, epoch, model, tracker or type('obj', (object,), {'save_model': lambda *args: None})())

                # Legacy Visdom support
                if cfg.VISDOM:
                    cfg.vis.line(X=torch.Tensor([epoch+1]).unsqueeze(0).cpu(),Y=torch.Tensor([val_loss]).unsqueeze(0).cpu(),env='torch',win=cfg.loss_window,name='val_loss',update='append')
                    cfg.vis.line(X=torch.Tensor([epoch+1]).unsqueeze(0).cpu(),Y=torch.Tensor([acc/100]).unsqueeze(0).cpu(),env='torch',win=cfg.loss_window,name='val_acc',update='append')

            # Save checkpoint periodically
            if cfg.SAVE_CHECKPOINT_FREQ > 0 and epoch % cfg.SAVE_CHECKPOINT_FREQ == 0 and epoch > 0:
                if tracker:
                    tracker.save_model(model, f"checkpoint_epoch_{epoch}.pth")

        end = (time.time() - start)//60
        logger.info(f"Training completed in: {end//1440}D {(end%1440)//60}H {end%60}M")

        # Log best model info
        if best_model_tracker:
            logger.info(f"Best {best_model_tracker.metric_name}: {best_model_tracker.best_value:.4f} at epoch {best_model_tracker.best_epoch}")

    # Final validation
    final_val_loss, final_acc = validate(val_loader, model, criterion, cfg)
    logger.info(f"Final validation - Loss: {final_val_loss:.4f}, Acc: {final_acc:.2f}%")

    # Uncertainty estimation
    if cfg.USE_BAYESIAN and cfg.ESTIMATE_UNCERTAINTY:
        logger.info("Estimating uncertainty on validation set...")
        uncertainty_estimator = UncertaintyEstimator(
            model, num_samples=cfg.MC_SAMPLES, device=cfg.DEVICE
        )

        uncertainty_results = uncertainty_estimator.estimate_uncertainty_batch(val_loader)

        # Analyze and visualize
        save_path = f"{cfg.OUTPUT_DIR}/experiments/{cfg.EXPERIMENT_NAME}/uncertainty_analysis.png" if tracker else None
        analysis = analyze_uncertainty(uncertainty_results, save_path=save_path)

        # Log to tracker
        if tracker:
            tracker.log_metrics({
                'ece': analysis['expected_calibration_error'],
                'mean_epistemic_unc': analysis['mean_epistemic_uncertainty'],
                'mean_total_unc': analysis['mean_total_uncertainty'],
            }, 0, prefix='uncertainty/')

        logger.info("Uncertainty estimation complete!")

    # Close tracker
    if tracker:
        tracker.finish()