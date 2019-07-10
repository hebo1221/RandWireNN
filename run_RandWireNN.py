import torch
from RandWireNN_train import train, validate, prepare
from utils.network import Net
from utils.config_helpers import merge_configs
from utils.dataloader import train_data_loader, val_data_loader
import time

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
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(cfg.DEVICE)
    
    criterion = torch.nn.CrossEntropyLoss().to(cfg.DEVICE)
    optimizer = torch.optim.SGD(model.parameters(),cfg.LEARNING_RATE, cfg.MOMENTUM, cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.EPOCH)

    if cfg.LOAD_TRAINED_MODEL:
        model.load_state_dict(torch.load(cfg.TRAINED_MODEL_LOAD_DIR))

    if not cfg.TEST_MODE:
        start = time.time()
        for epoch in range(cfg.EPOCH+1):
            train(train_loader, model, criterion, optimizer, epoch, cfg)
            scheduler.step()
            if epoch % cfg.VAL_FREQ == 0:
                val_loss, acc = validate(val_loader, model, criterion, cfg)
                if cfg.VISDOM:
                    cfg.vis.line(X=torch.Tensor([epoch+1]).unsqueeze(0).cpu(),Y=torch.Tensor([val_loss]).unsqueeze(0).cpu(),env='torch',win=cfg.loss_window,name='val_loss',update='append')      
                    cfg.vis.line(X=torch.Tensor([epoch+1]).unsqueeze(0).cpu(),Y=torch.Tensor([acc/100]).unsqueeze(0).cpu(),env='torch',win=cfg.loss_window,name='val_acc',update='append')      
        end = (time.time() - start)//60
        print("train time: {}D {}H {}M".format(end//1440, (end%1440)//60, end%60))

    validate(val_loader, model, criterion, cfg)