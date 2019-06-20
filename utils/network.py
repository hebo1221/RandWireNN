# referred to JiaminRen's implementation
# https://github.com/JiaminRen/RandWireNN

import torch
import torch.nn as nn
from utils.graph import build_graph, get_graph_info, save_graph, load_graph
import math
import torch.nn.functional as F

class conv_unit(nn.Module):
    def __init__(self, nin, nout, stride):
        super(conv_unit, self).__init__()
        self.depthwise_separable_conv_3x3 = nn.Conv2d(nin, nin, kernel_size=3, stride=stride, padding=1, groups=nin)
        self.pointwise_conv_1x1 = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise_separable_conv_3x3(x)
        out = self.pointwise_conv_1x1(out)
        return out

class Triplet_unit(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(Triplet_unit, self).__init__()
        self.relu = nn.ReLU()
        self.conv = conv_unit(inplanes, outplanes, stride)
        self.bn = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv(out)
        out = self.bn(out)
        return out

class Node_OP(nn.Module):
    def __init__(self, Node, inplanes, outplanes):
        super(Node_OP, self).__init__()
        self.is_input_node = Node.type == 0
        self.input_nums = len(Node.inputs)
        if self.input_nums > 1:
            self.mean_weight = nn.Parameter(torch.ones(self.input_nums))
            self.sigmoid = nn.Sigmoid()
        if self.is_input_node:
            self.conv = Triplet_unit(inplanes, outplanes, stride=2)
        else:
            self.conv = Triplet_unit(outplanes, outplanes, stride=1)

    def forward(self, *input):
        if self.input_nums > 1:
            out = self.sigmoid(self.mean_weight[0]) * input[0]
        for i in range(1, self.input_nums):
            out = out + self.sigmoid(self.mean_weight[i]) * input[i]
        else:
            out = input[0]
            out = self.conv(out)
        return out

class StageBlock(nn.Module):
    def __init__(self, graph, inplanes, outplanes):
        super(StageBlock, self).__init__()
        self.nodes, self.input_nodes, self.output_nodes = get_graph_info(graph)
        self.nodeop  = nn.ModuleList()
        for node in self.nodes:
            self.nodeop.append(Node_OP(node, inplanes, outplanes))

    def forward(self, x):
        results = {}
        for id in self.input_nodes:
            results[id] = self.nodeop[id](x)
        for id, node in enumerate(self.nodes):
            if id not in self.input_nodes:
                results[id] = self.nodeop[id](*[results[_id] for _id in node.inputs])
        result = results[self.output_nodes[0]]
        for idx, id in enumerate(self.output_nodes):
            if idx > 0:
                result = result + results[id]
        result = result / len(self.output_nodes)
        return result
    
class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        # for image color scale
        color = cfg.NN.COLOR
        N = cfg.NN.NODES
        C = cfg.NN.CHANNELS
        size = cfg.NN.IMG_SIZE
        num_classes = cfg.NN.NUM_CLASSES

        if cfg.MAKE_GRAPH:
            graph2 = build_graph(N//2, cfg)
            graph3 = build_graph(N, cfg)
            graph4 = build_graph(N, cfg)
            graph5 = build_graph(N, cfg)
            save_graph(graph2, './output/graph/conv2.yaml')
            save_graph(graph3, './output/graph/conv3.yaml')
            save_graph(graph4, './output/graph/conv4.yaml')
            save_graph(graph5, './output/graph/conv5.yaml')
        else:
            graph2 = load_graph('./output/graph/conv2.yaml')
            graph3 = load_graph('./output/graph/conv3.yaml')
            graph4 = load_graph('./output/graph/conv4.yaml')
            graph5 = load_graph('./output/graph/conv5.yaml')
        
        if cfg.NN.REGIME == "SMALL":
    
            self.conv1 =  nn.Sequential(
                conv_unit(color, C//2, 2),
                nn.BatchNorm2d(C//2)
                )
            self.conv2 = Triplet_unit(C//2, C)
            self.conv3 = StageBlock(graph3, C, C)
            self.conv4 = StageBlock(graph4, C, 2*C)
            self.conv5 = StageBlock(graph5, 2*C, 4*C)
            self.classifier = nn.Sequential(
                    nn.ReLU(True),
                    nn.Conv2d(4*C, 1280, kernel_size=1),
                    nn.BatchNorm2d(1280),
                    nn.ReLU(True),
                    nn.AvgPool2d(size//16, stride=1),
                )
            self.fc = nn.Linear(1280, num_classes)

        if cfg.NN.REGIME == "REGULAR":
            self.conv1 =  nn.Sequential(
                conv_unit(color, C//2, 2),
                nn.BatchNorm2d(C//2)
                )
            self.conv2 = StageBlock(graph2, C//2,  C)
            self.conv3 = StageBlock(graph3, C,   2*C)
            self.conv4 = StageBlock(graph4, 2*C, 4*C)
            self.conv5 = StageBlock(graph5, 4*C, 8*C)
            self.classifier = nn.Sequential(
                    nn.ReLU(True),
                    nn.Dropout(0.2),
                    nn.Conv2d(8*C, 1280, kernel_size=1),
                    nn.Dropout(0.2),
                    nn.BatchNorm2d(1280),
                    nn.ReLU(True),
                    nn.AvgPool2d(size//8, stride=1),
                )
            self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x