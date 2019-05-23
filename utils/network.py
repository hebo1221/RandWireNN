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

class CNN(nn.Module):
    img_size = 0
    def __init__(self, cfg):
        super(CNN, self).__init__()
        color = cfg.NN.COLOR
        nodes = cfg.NN.NODES
        channels = cfg.NN.CHANNELS
        num_classes = cfg.NN.NUM_CLASSES
        img_size = cfg.NN.IMG_SIZE
        
        self.conv1 = nn.Conv2d(color, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.img_size = int((((img_size-4)/2)-4)/2)**2
        self.fc1 = nn.Linear(16 * self.img_size , channels*2)
        self.fc2 = nn.Linear(channels*2, channels)
        self.fc3 = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.img_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net(nn.Module):
  def __init__(self, cfg):
    super(Net, self).__init__()
    nodes = cfg.NN.NODES
    channels = cfg.NN.CHANNELS
    num_classes = cfg.NN.NUM_CLASSES
    seed = 0
    
    self.conv1 =  nn.Sequential(
        conv_unit(3, channels // 2 , 2),
        nn.BatchNorm2d(channels // 2)
        )
    self.conv2 = Triplet_unit(channels // 2, channels, 2)
    
    graph = load_graph('./output/graph/conv3.yaml')
    #graph = build_graph(nodes, 'WS', seed, 4, 0.75)
    #save_graph(graph, './output/graph/conv3.yaml')
    self.conv3 = StageBlock(graph, channels, channels)
    
    graph = load_graph('./output/graph/conv4.yaml')
    #graph = build_graph(nodes, 'WS', seed, 4, 0.75)
    #save_graph(graph, './output/graph/conv4.yaml')
    self.conv4 = StageBlock(graph, channels, channels *2)
    
    graph =load_graph('./output/graph/conv5.yaml')
    #graph = build_graph(nodes, 'WS', seed, 4, 0.75)
    #save_graph(graph, './output/graph/conv5.yaml')
    self.conv5 = StageBlock(graph, channels * 2, channels * 4)

    self.classifier = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(channels * 4, 1280, kernel_size=1),
            nn.BatchNorm2d(1280),
            nn.ReLU(True),
            nn.AvgPool2d(7, stride=1),
        )

    self.fc = nn.Linear(1280, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))

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
