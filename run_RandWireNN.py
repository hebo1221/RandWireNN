import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from RandWireNN_train import train,  prepare
from utils.network import CNN
from utils.config_helpers import merge_configs

from CVdevKit.dataset.imagenet_dataset import ColorAugmentation, ImagenetDataset
import time
import os
import numpy as np

def get_configuration():
    # load configs for base network, random graph model and data set
    from RandWireNN_config import cfg as network_cfg
    from utils.configs.WS_config import cfg as graph_cfg
    # from utils.configs.BA_config import cfg as graph_cfg
    # from utils.configs.WS_config import cfg as graph_cfg
    from utils.configs.CIFAR100 import cfg as dataset_cfg
    # for the CIFAR10 data set use:     from utils.configs.ImageNet_config import cfg as dataset_cfg
    # for the ImageNet data set use:     from utils.configs.Grocery_config import cfg as dataset_cfg
    return merge_configs([network_cfg, graph_cfg, dataset_cfg])


if __name__ == '__main__':

    cfg = get_configuration()
    prepare(cfg)

    train_root = "G:/dataset/ILSVRC2012_img_train/" 

    model = CNN(32, 79)

    # load
    model.load_state_dict(torch.load("./output/model/054_000000.cpt"))
    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),lr=0.005, momentum=0.9, weight_decay=0.00005)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("train_loader")
    train_loader = DataLoader(
                datasets.ImageFolder(train_root, transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=128, 
                shuffle=True, pin_memory=True, drop_last=True)

    
    for epoch in range(54,100):
        print("train")
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
 