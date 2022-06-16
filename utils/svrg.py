from torch.optim import Optimizer
import copy
import torch

class SVRG_k(Optimizer):
    r"""Optimization class for calculating the gradient of one iteration.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """
    def __init__(self, params, lr):
        #print("Using optimizer: SVRG")
        self.u = None
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(SVRG_k, self).__init__(params, defaults)
    
    def get_param_groups(self):
            return self.param_groups

    def set_u(self, new_u):
        """Set the mean gradient for the current epoch. 
        """
        if self.u is None:
            self.u = copy.deepcopy(new_u)
        for u_group, new_group in zip(self.u, new_u):  
            for u, new_u in zip(u_group['params'], new_group['params']):
                u.grad = new_u.grad.clone()

    def step(self, params):
        """Performs a single optimization step.
        """
        for group, new_group, u_group in zip(self.param_groups, params, self.u):
            for p, q, u in zip(group['params'], new_group['params'], u_group['params']):
                if p.grad is None:
                    continue
                if q.grad is None:
                    continue
                # core SVRG gradient update 
                new_d = p.grad.data - q.grad.data + u.grad.data
                #p=p-group["lr"]*new_d
                p.data.add_(new_d, alpha=-group['lr'])


class SVRG_Snapshot(Optimizer):
    r"""Optimization class for calculating the mean gradient (snapshot) of all samples.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """
    def __init__(self, params):
        defaults = dict()
        super(SVRG_Snapshot, self).__init__(params, defaults)
      
    def get_param_groups(self,batch=1):
        batch=float(batch)
        for i in self.param_groups[0]['params']:
            
            i.grad=torch.div(i.grad,batch)
        return self.param_groups
    
    def set_param_groups(self, new_params):
        """Copies the parameters from another optimizer. 
        """
        for group, new_group in zip(self.param_groups, new_params): 
            for p, q in zip(group['params'], new_group['params']):
                  p.data[:] = q.data[:]