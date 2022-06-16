#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
from numpy import dtype
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvgP(w,args):
    w_avg = torch.zeros(w[0].shape[0], dtype=w[0].dtype, device=args.device)
    for k in w:
        w_avg+=k
    w_avg = torch.div(w_avg, len(w)).detach()
    return w_avg

def FedAvgGradient(grads_list):
    num_client=float(len(grads_list))

    for p0 in grads_list[0][0]['params']:
        p0.grad=torch.div(p0.grad,num_client)
    for i in range(1,len(grads_list)):
        for p0,para in zip(grads_list[0][0]['params'],grads_list[i][0]['params']):
            p0.grad=p0.grad+torch.div(para.grad,num_client)
    # for p0 in grads_list[0][0]['params']:
    #     print(p0.grad)     
    return grads_list[0]
