import torch
from RandWireNN_train import train, validate, prepare
from utils.network import CNN, Net
from utils.config_helpers import merge_configs
from utils.dataloader import train_data_loader, val_data_loader

def get_configuration():
    # load configs for base network and data set
    from RandWireNN_config import cfg as network_cfg

    from utils.configs.cifar100_config import cfg as dataset_cfg
    # for the CIFAR10 data set use:     from utils.configs.cifar10 import cfg as dataset_cfg
    # for the ImageNet data set use:    from utils.configs.ImageNet_config import cfg as dataset_cfg
    
    return merge_configs([network_cfg, dataset_cfg])


if __name__ == '__main__':
    cfg = get_configuration()
    prepare(cfg)

    if cfg.USE_SIMPLE_CNN:
        model = CNN(cfg)
    else:
        model = Net(cfg)
    
    if cfg.LOAD_TRAINED_MODEL:
        model.load_state_dict(torch.load(cfg.TRAINED_MODEL_LOAD_DIR))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),cfg.LEARNING_RATE, cfg.MOMENTUM, cfg.WEIGHT_DECAY)

    if cfg.USE_CUDA:
        model.cuda()
        criterion.cuda()

    print("train_loader")
    train_loader = train_data_loader(cfg)
    print("val_loader")
    val_loader = val_data_loader(cfg)

    print("train")
    #for epoch in range(cfg.EPOCH):
        #train(train_loader, model, criterion, optimizer, epoch, cfg)
    
    validate(val_loader, model, criterion, cfg)