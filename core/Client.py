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
        self.net = net

        self.ldr_train = DataLoader(DatasetSplit(
            dataset, idxs[client_id]), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_val = DataLoader(DatasetSplit(
            dataset, idxs[-client_id-1]), batch_size=self.args.local_bs, shuffle=True)

        self.hyper_param=copy.deepcopy(hyper_param)
        self.hyper_param_init=copy.deepcopy(hyper_param)
        self.hyper_optimizer= SGD([self.hyper_param[k] for k in self.hyper_param],
                                lr=args.hlr)
        self.val_loss = self.cross_entropy
        self.loss_func = self.loss_adjust_cross_entropy #nn.CrossEntropyLoss()
        self.hyper_iter = 0

    def train_epoch(self):
        pass

    def batch_grad(self):
        self.net0 = copy.deepcopy(self.net)
        self.net0.train()
        optimizer = SVRG_Snapshot(self.net0.parameters())
        self.net0.zero_grad()
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)
            log_probs = self.net0(images)
            loss = self.loss_func(log_probs, labels)
            loss.backward()
        return optimizer.get_param_groups(batch_idx+1)

    def grad_d_in_d_y(self):
        self.net0 = copy.deepcopy(self.net)
        self.net0.train()
        num_weights = sum(p.numel() for p in self.net0.parameters())
        d_in_d_y = torch.zeros(num_weights, device=self.args.device)
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)
            self.net0.zero_grad()
            log_probs = self.net0(images)
            loss = self.loss_func(log_probs, labels)
            d_in_d_y += gather_flat_grad(grad(loss,
                                         self.net0.parameters(), create_graph=True))
        d_in_d_y /= (batch_idx+1.)
        return d_in_d_y

    def grad_d_out_d_y(self):
        self.net0 = copy.deepcopy(self.net)
        self.net0.train()
        num_weights = sum(p.numel() for p in self.net0.parameters())
        d_out_d_y = torch.zeros(num_weights, device=self.args.device)
        for batch_idx, (images, labels) in enumerate(self.ldr_val):
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)
            self.net0.zero_grad()
            log_probs = self.net0(images)
            loss = self.val_loss(log_probs, labels)
            d_out_d_y += gather_flat_grad(grad(loss,
                                         self.net0.parameters(), create_graph=True))
        d_out_d_y /= (batch_idx+1.)
        return d_out_d_y


    def hvp_iter(self, p, lr):
        if self.hyper_iter == 0:
            self.d_in_d_y = self.grad_d_in_d_y()
            self.counter = p.clone()
        old_counter = self.counter
        hessian_term = gather_flat_grad(
            grad(self.d_in_d_y, self.net0.parameters(),
                 grad_outputs=self.counter.view(-1), retain_graph=True)
        )
        self.counter = old_counter - lr * hessian_term
        p = p+self.counter
        self.hyper_iter += 1
        return p
    
    def grad_d_out_d_x(self, hyper_param = None):
        if hyper_param == None:
            hyper_param == self.hyper_param
        self.net0 = copy.deepcopy(self.net)
        self.net0.train()
        num_weights = sum(p.numel() for p in self.net0.parameters())
        d_out_d_x = torch.zeros(num_weights, device=self.args.device)
        for batch_idx, (images, labels) in enumerate(self.ldr_val):
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)
            self.net0.zero_grad()
            log_probs = self.net0(images)
            loss = self.val_loss(log_probs, labels, hyper_param = hyper_param)
            d_out_d_x += gather_flat_grad(grad(loss,
                                         get_trainable_hyper_params(hyper_param), create_graph=True))
        d_out_d_x /= (batch_idx+1.)
        return d_out_d_x

    def hyper_grad(self, p):
        d_in_d_y=self.grad_d_in_d_y()
        indirect_grad= gather_flat_grad(
            grad(d_in_d_y,
                get_trainable_hyper_params(self.hyper_param),
                grad_outputs= p.view(-1),
                allow_unused= True)
        )
        try:
            direct_grad= self.grad_d_out_d_x()
            hyper_grad=direct_grad-indirect_grad
        except:
            hyper_grad=-indirect_grad
        return hyper_grad

    def hyper_update(self, hg):
        assign_hyper_gradient(self.hyper_param, hg)
        self.hyper_optimizer.step()
        return -gather_flat_hyper_params(self.hyper_param)+gather_flat_hyper_params(self.hyper_param_init)
    
    def hyper_svrg_update(self, hg):
        try:
            direct_grad = self.grad_d_out_d_x()
            direct_grad_0 = self.grad_d_out_d_x(hyper_param=self.hyper_param_init)
            h = direct_grad - direct_grad_0 + hg
        except:
            h = hg
        assign_hyper_gradient(self.hyper_param, h)
        self.hyper_optimizer.step()
        return -gather_flat_hyper_params(self.hyper_param)+gather_flat_hyper_params(self.hyper_param_init)



    def loss_adjust_cross_entropy(self, logits, targets):
        dy = self.hyper_param['dy']
        ly = self.hyper_param['ly']
        x = logits*torch.sigmoid(dy)+ly
        loss = F.cross_entropy(x, targets)
        return loss
    
    def cross_entropy(self, logits, targets, hyper_param = None):
        return F.cross_entropy(logits, targets, weight=self.hyper_param['wy'])



