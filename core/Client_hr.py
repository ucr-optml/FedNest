import copy
from math import ceil
from warnings import catch_warnings
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from core.function import gather_flat_grad, get_trainable_hyper_params, loss_adjust_cross_entropy, gather_flat_hyper_params
from utils.svrg import SVRG_Snapshot
from numpy import random
from torch.autograd import grad
import torch.nn.functional as F
from torch.optim import SGD
from core.function import assign_hyper_gradient

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Client():
    def __init__(self, args, client_id, net, dataset=None, idxs=None, hyper_param = None) -> None:
        self.client_id = client_id
        self.args = args
        self.net = copy.deepcopy(net)
        self.init_net = copy.deepcopy(net)
        self.net.zero_grad()
        self.init_net.zero_grad()
        self.beta = 0.1

        self.ldr_train = DataLoader(DatasetSplit(
            dataset, idxs[client_id]), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_val = DataLoader(DatasetSplit(
            dataset, idxs[-client_id-1]), batch_size=self.args.local_bs, shuffle=True)

        self.hyper_param= [k for n,k in self.net.named_parameters() if not "header" in n]
        self.hyper_param_init = [k for n,k in self.init_net.named_parameters() if not "header" in n]
        self.hyper_optimizer= SGD(self.hyper_param,
                                lr=args.hlr)
        self.val_loss = self.cross_entropy
        self.loss_func = self.cross_entropy_reg #nn.CrossEntropyLoss()
        self.hyper_iter = 0

    def train_epoch(self):
        pass

    def batch_grad(self):
        self.net0 = copy.deepcopy(self.net)
        self.net0.train()
        optimizer = SVRG_Snapshot([k for k in self.net0.parameters() if k.requires_grad==True])
        self.net0.zero_grad()
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)
            log_probs = self.net0(images)
            loss = self.loss_func(log_probs, labels, [k for k in self.net0.parameters() if k.requires_grad==True])
            loss.backward()
        return optimizer.get_param_groups(batch_idx+1)

    def grad_d_in_d_y(self):
        self.net0 = copy.deepcopy(self.net)
        self.net0.train()
        hyper_param = [k for n,k in self.net0.named_parameters() if not "header" in n]
        params = [k for n,k in self.net0.named_parameters() if "header" in n]
        num_weights = sum(p.numel() for p in params)
        d_in_d_y = torch.zeros(num_weights, device=self.args.device)
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)
            self.net0.zero_grad()
            log_probs = self.net0(images)
            loss = self.loss_func(log_probs, labels, params)
            d_in_d_y += gather_flat_grad(grad(loss,
                                         params, create_graph=True))
        d_in_d_y /= (batch_idx+1.)
        return d_in_d_y, hyper_param

    def grad_d_out_d_y(self):
        self.net0 = copy.deepcopy(self.net)
        self.net0.train()
        hyper_param = [k for n,k in self.net0.named_parameters() if not "header" in n]
        params = [k for n,k in self.net0.named_parameters() if "header" in n]
        num_weights = sum(p.numel() for p in params)
        d_out_d_y = torch.zeros(num_weights, device=self.args.device)
        for batch_idx, (images, labels) in enumerate(self.ldr_val):
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)
            self.net0.zero_grad()
            log_probs = self.net0(images)
            loss = self.val_loss(log_probs, labels)
            d_out_d_y += gather_flat_grad(grad(loss,
                                         params, create_graph=True))
        d_out_d_y /= (batch_idx+1.)
        return d_out_d_y, hyper_param


    def hvp_iter(self, p, lr):
        if self.hyper_iter == 0:
            self.d_in_d_y,_ = self.grad_d_in_d_y()
            self.counter = p.clone()
        params = [k for n,k in self.net0.named_parameters() if "header" in n]
        old_counter = self.counter
        hessian_term = gather_flat_grad(
            grad(self.d_in_d_y, params,
                 grad_outputs=self.counter.view(-1), retain_graph=True)
        )
        self.counter = old_counter - lr * hessian_term
        p = p+self.counter
        self.hyper_iter += 1
        return p
    
    def grad_d_out_d_x(self, net = None):
        if net == None:
            net = copy.deepcopy(self.net)
        else:
            net = copy.deepcopy(net)
        net.train()
        hyper_param = [k for n,k in net.named_parameters() if not "header" in n]
        num_weights = sum(p.numel() for p in hyper_param)

        d_out_d_x = torch.zeros(num_weights, device=self.args.device)
        for batch_idx, (images, labels) in enumerate(self.ldr_val):
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)
            net.zero_grad()
            log_probs = net(images)
            loss = self.val_loss(log_probs, labels)
            d_out_d_x += gather_flat_grad(grad(loss,
                                         get_trainable_hyper_params(hyper_param), create_graph=True))
        d_out_d_x /= (batch_idx+1.)
        return d_out_d_x

    def hyper_grad(self, p):
        d_in_d_y, hyper_param=self.grad_d_in_d_y()
        indirect_grad= gather_flat_grad(
            grad(d_in_d_y,
                get_trainable_hyper_params(hyper_param),
                grad_outputs= p.view(-1),
                allow_unused= True)
        )
        try:
            direct_grad= self.grad_d_out_d_x()
            hyper_grad=direct_grad-self.args.hlr*indirect_grad
        except:
            #print(" No direct grad, use only indirect gradient.")
            hyper_grad=-indirect_grad
        return hyper_grad

    def hyper_update(self, hg):

        assign_hyper_gradient(self.hyper_param, hg.detach())
        self.hyper_optimizer.step()
        return -gather_flat_hyper_params(self.hyper_param)+gather_flat_hyper_params(self.hyper_param_init)
    
    def hyper_svrg_update(self, hg):
        try:
            direct_grad = self.grad_d_out_d_x()
            direct_grad_0 = self.grad_d_out_d_x(net=self.init_net)
            h = direct_grad - direct_grad_0 + hg
        except:
            h = hg
        h=h.detach()
        assign_hyper_gradient(self.hyper_param, h)
        self.hyper_optimizer.step()
        return -gather_flat_hyper_params(self.hyper_param)+gather_flat_hyper_params(self.hyper_param_init)
    
    def cross_entropy(self, logits, targets):
        return F.cross_entropy(logits, targets)
        
    def cross_entropy_reg(self, logits, targets, param):
        reg = self.beta*sum([torch.norm(k) for k in param])
        return F.cross_entropy(logits, targets)+0.5*reg



