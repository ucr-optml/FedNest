#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from utils.svrg import SVRG_k,SVRG_Snapshot
import copy

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, avg_q):
        net.train()
        # train and update
        optimizer = SVRG_k(net.parameters(), lr=self.args.lr)
        optimizer.set_u(avg_q)

        net0 = copy.deepcopy(net)
        net0.train()
        optim0 = SVRG_Snapshot(net0.parameters())

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                net0.zero_grad()
                log_probs_0=net0(images)
                loss0=self.loss_func(log_probs_0, labels)
                loss0.backward()
                param_group=optim0.get_param_groups(1)

                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step(param_group)
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def batch_grad(self, net):
        net.train()
        optimizer = SVRG_Snapshot(net.parameters())
        net.zero_grad()
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            loss.backward()
        return optimizer.get_param_groups(batch_idx+1)

