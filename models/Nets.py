#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from importlib_metadata import requires
from numpy import dtype
import torch
from torch import nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, d, n):
        super(Linear, self).__init__()
        self.y_inner = torch.ones(n, dtype=torch.float32)*10
        self.x_outer = torch.ones( d, dtype=torch.float32)*10
        self.y_inner.requires_grad=True
        self.x_outer.requires_grad=True
        self.y_inner = nn.Parameter(self.y_inner)
        self.x_outer = nn.Parameter(self.x_outer)
    def forward(self, A):
        #y_square = -0.5* torch.t(self.y_header) * self.y_header
        return torch.matmul(A,self.x_outer)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.header = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.header(x)
        return x

# class MLP(nn.Module):
#     def __init__(self, dim_in, dim_hidden, dim_out):
#         super(MLP, self).__init__()
#         self.layer_input = nn.Linear(dim_in, dim_hidden)
#         self.layer_mid = nn.Linear(dim_hidden, 100)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout()
#         self.header = nn.Linear(100, dim_out)

#     def forward(self, x):
#         x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
#         x = self.layer_input(x)
#         x = self.dropout(x)
#         x = self.relu(x)
#         x = self.layer_mid(x)
#         x = self.relu(x)
#         x = self.header(x)
#         return x

# class CNNMnist(nn.Module):
#     def __init__(self, args):
#         super(CNNMnist, self).__init__()
#         self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fC1_outer = nn.Linear(320, 50)
#         self.header = nn.Linear(50, args.num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fC1_outer(x))
#         x = F.dropout(x, training=self.training)
#         x = self.header(x)
#         return x

class CNNMnist(nn.Module):

    def __init__(self,args):
        super(CNNMnist, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fC1_outer = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.header1 = nn.Linear(120, 84)
        self.header2 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fC1_outer(x))
        x = F.relu(self.header1(x))
        x = self.header2(x)
        return x

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fC1_outer = nn.Linear(16 * 5 * 5, 120)
        self.fC2_outer = nn.Linear(120, 84)
        self.header = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fC1_outer(x))
        x = F.relu(self.fC2_outer(x))
        x = self.header(x)
        return x

from torch.autograd import Variable as V

class Weights:
     def __init__(self, C1_outer_, C2_outer_, F1_outer_, F2_outer_, BC1_outer_, BC2_outer_, BF1_outer_, BF2_outer_):
          self.C1_outer = C1_outer_
          self.C2_outer = C2_outer_
          self.F1_outer = F1_outer_
          self.F2_outer = F2_outer_
          self.BC1_outer = BC1_outer_
          self.BC2_outer = BC2_outer_
          self.BF1_outer = BF1_outer_
          self.BF2_outer = BF2_outer_

class MM_CNN(nn.Module):
    def __init__(self, args):
        super(MM_CNN, self).__init__()
        self.C1_outer = V(torch.zeros(5, 1, 3, 3), requires_grad=True)
        self.C2_outer = V(torch.zeros(10, 5, 3, 3), requires_grad=True)
        self.F1_outer = V(torch.zeros(5 * 5 * 10, 100), requires_grad=True)
        self.F2_outer = V(torch.zeros(100, 10), requires_grad=True)
        torch.nn.init.xavier_normal_(self.C1_outer.data)
        torch.nn.init.xavier_normal_(self.C2_outer.data)
        torch.nn.init.xavier_normal_(self.F1_outer.data)
        torch.nn.init.xavier_normal_(self.F2_outer.data)
        self.BC1_outer = V(torch.randn(5) * 1 / 8, requires_grad=True)
        self.BC2_outer = V(torch.randn(10) * 1 / 16, requires_grad=True)
        self.BF1_outer = V(torch.randn(100) * 1 / 100, requires_grad=True)
        self.BF2_outer = V(torch.randn(10) * 1 / 10, requires_grad=True)
        #w = Weights(C1_outer.data, C2_outer.data, F1_outer.data, F2_outer.data, BC1_outer.data, BC2_outer.data, BF1_outer.data, BF2_outer.data)
        self.C1_outer = nn.Parameter(self.C1_outer)
        self.C2_outer = nn.Parameter(self.C2_outer)
        self.F1_outer = nn.Parameter(self.F1_outer)
        self.F2_outer = nn.Parameter(self.F2_outer)
        self.BC1_outer = nn.Parameter(self.BC1_outer)
        self.BC2_outer = nn.Parameter(self.BC2_outer)
        self.BF1_outer = nn.Parameter(self.BF1_outer)
        self.BF2_outer = nn.Parameter(self.BF2_outer)
        self.t_inner = V(torch.ones(10), requires_grad=True)
        self.t_inner = nn.Parameter(self.t_inner)
    def forward(self,x):
        batch_size=x.shape[0]
        x = F.conv2d(x, self.C1_outer, bias=self.BC1_outer)
        x = torch.tanh(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.conv2d(x, self.C2_outer, bias=self.BC2_outer)
        x = torch.tanh(x)
        x = F.max_pool2d(x, 2, 2)
        x = torch.tanh(x.view(batch_size, 5 * 5 * 10).mm(self.F1_outer) + self.BF1_outer)
        pred = x.mm(self.F2_outer) + self.BF2_outer
        return pred