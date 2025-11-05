import torch
from RandWireNN_train import train, validate, prepare
from utils.network import Net
from utils.config_helpers import merge_configs
from utils.dataloader import train_data_loader, val_data_loader
from utils.optimizers import get_optimizer, get_scheduler
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

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model.to(cfg.DEVICE)

    criterion = torch.nn.CrossEntropyLoss().to(cfg.DEVICE)

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

    if cfg.LOAD_TRAINED_MODEL:
        model.load_state_dict(torch.load(cfg.TRAINED_MODEL_LOAD_DIR))

    if not cfg.TEST_MODE:
        start = time.time()
        for epoch in range(cfg.EPOCH+1):
            # Pass scheduler to train function for OneCycleLR
            train(train_loader, model, criterion, optimizer, epoch, cfg, scaler=scaler, scheduler=scheduler)

            # Step scheduler (handle different scheduler types)
            if cfg.SCHEDULER.lower() == 'reduce_on_plateau':
                # ReduceLROnPlateau needs validation loss
                val_loss, acc = validate(val_loader, model, criterion, cfg)
                if scheduler is not None:
                    scheduler.step(val_loss)
            elif cfg.SCHEDULER.lower() != 'onecycle' and scheduler is not None:
                # OneCycleLR steps per batch, not per epoch
                scheduler.step()

            if epoch % cfg.VAL_FREQ == 0:
                val_loss, acc = validate(val_loader, model, criterion, cfg)
                if cfg.VISDOM:
                    cfg.vis.line(X=torch.Tensor([epoch+1]).unsqueeze(0).cpu(),Y=torch.Tensor([val_loss]).unsqueeze(0).cpu(),env='torch',win=cfg.loss_window,name='val_loss',update='append')
                    cfg.vis.line(X=torch.Tensor([epoch+1]).unsqueeze(0).cpu(),Y=torch.Tensor([acc/100]).unsqueeze(0).cpu(),env='torch',win=cfg.loss_window,name='val_acc',update='append')
        end = (time.time() - start)//60
        logger.info(f"Training completed in: {end//1440}D {(end%1440)//60}H {end%60}M")

    validate(val_loader, model, criterion, cfg)