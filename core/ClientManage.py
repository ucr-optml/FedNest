import copy
from cv2 import log
import numpy as np

import torch


from utils.Fed import FedAvg,FedAvgGradient, FedAvgP
from core.SGDClient import SGDClient
from core.SVRGClient import SVRGClient
from core.Client import Client

class ClientManage():
    def __init__(self,args, net_glob, client_idx, dataset, dict_users, hyper_param) -> None:
        self.net_glob=net_glob
        self.client_idx=client_idx
        self.args=args
        self.dataset=dataset
        self.dict_users=dict_users
           
        self.hyper_param = copy.deepcopy(hyper_param)

    def fed_in(self):
        print(self.client_idx)
        w_glob = self.net_glob.state_dict()
        if self.args.all_clients:
            print("Aggregation over all clients")
            w_locals = [w_glob for i in range(self.args.num_users)]
        else:
            w_locals=[]

        loss_locals = []
        grad_locals = []
        client_locals = []

        for idx in self.client_idx:
            if self.args.optim == 'sgd':
                client = SGDClient(self.args, idx, copy.deepcopy(self.net_glob),self.dataset, self.dict_users, self.hyper_param)
            elif self.args.optim == 'svrg':
                client = SVRGClient(self.args, idx, copy.deepcopy(self.net_glob),self.dataset, self.dict_users, self.hyper_param)
                grad = client.batch_grad()
                grad_locals.append(grad)
            else:
                raise NotImplementedError
            client_locals.append(client)
        if self.args.optim == 'svrg':
            avg_grad = FedAvgGradient(grad_locals)
            for client in client_locals:
                client.set_avg_q(avg_grad)
        for client in client_locals:
            w, loss = client.train_epoch()
            if self.args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        self.net_glob.load_state_dict(w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)
        return w_glob, loss_avg

    def fedIHGP(self,client_locals):
        d_out_d_y_locals=[]
        for client in client_locals:
            d_out_d_y=client.grad_d_out_d_y()
            d_out_d_y_locals.append(d_out_d_y)
        p=FedAvgP(d_out_d_y_locals,self.args)
        
        p_locals=[]
        if self.args.hvp_method == 'global_batch':
            for i in range(self.args.neumann):
                for client in client_locals:
                    p_client = client.hvp_iter(p, self.args.hlr)
                    p_locals.append(p_client)
                p=FedAvgP(p_locals, self.args)
        elif self.args.hvp_method == 'local_batch':
            for client in client_locals:
                p_client=p.clone()
                for _ in range(self.args.neumann):
                    p_client = client.hvp_iter(p_client, self.args.hlr)
                p_locals.append(p_client)
            p=FedAvgP(p_locals, self.args)
        elif self.args.hvp_method == 'seperate':
            for client in client_locals:
                d_out_d_y=client.grad_d_out_d_y()
                p_client=d_out_d_y.clone()
                for _ in range(self.args.neumann):
                    p_client = client.hvp_iter(p_client, self.args.hlr)
                p_locals.append(p_client)
            p=FedAvgP(p_locals, self.args)

        else:
            raise NotImplementedError
        return p
    def lfed_out(self,client_locals):
        hg_locals =[]
        for client in client_locals:
            for _ in range(self.args.outer_tau):
                client.hyper_iter=0
                d_out_d_y=client.grad_d_out_d_y()
                p_client=d_out_d_y.clone()
                for _ in range(self.args.neumann):
                    p_client = client.hvp_iter(p_client, self.args.hlr)
                hg_client = client.hyper_grad(p_client.clone())
                hg = client.hyper_update(hg_client)
            hg_locals.append(hg)
        hg_glob=FedAvgP(hg_locals, self.args)
        return hg_glob, 1

    def fed_out(self):
        client_locals=[]
        for idx in self.client_idx:
            client= Client(self.args, idx, copy.deepcopy(self.net_glob),self.dataset, self.dict_users, self.hyper_param)
            client_locals.append(client)

        if self.args.hvp_method == 'seperate':
            return self.lfed_out(client_locals)

        #for client in client_locals:
        p = self.fedIHGP(client_locals)
        comm_round = 1+ self.args.neumann

        hg_locals =[]
        for client in client_locals:
            hg= client.hyper_grad(p.clone())
            hg_locals.append(hg)
        hg_glob=FedAvgP(hg_locals, self.args)
        comm_round+=1
        hg_locals =[]
        for client in client_locals:
            for _ in range(self.args.outer_tau):
                h = client.hyper_svrg_update(hg_glob)
            hg_locals.append(h)
            
        hg_glob=FedAvgP(hg_locals, self.args)
        comm_round+=1


        return hg_glob, comm_round

            


    
