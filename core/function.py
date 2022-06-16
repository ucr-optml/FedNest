from numpy import dtype
import torch.nn.functional as F
import torch
from torch.autograd import grad


def gather_flat_grad(loss_grad):
    # convert the gradient output from list of tensors to to flat vector 
    return torch.cat([p.contiguous().view(-1) for p in loss_grad if not p is None])


def neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr, num_neumann_terms, model):
    preconditioner = d_val_loss_d_theta.detach()
    counter = preconditioner
    # Do the fixed point iteration to approximate the vector-inverseHessian product
    i = 0
    while i < num_neumann_terms:  # for i in range(num_neumann_terms):
        old_counter = counter
        # This increments counter to counter * (I - hessian) = counter - counter * hessian
        hessian_term = gather_flat_grad(
            grad(d_train_loss_d_w, model.parameters(), grad_outputs=counter.view(-1), retain_graph=True))
        counter = old_counter - elementary_lr * hessian_term
        preconditioner = preconditioner + counter
        i += 1
    return elementary_lr * preconditioner


def loss_adjust_cross_entropy(logits, targets, params, group_size=1):
    # loss adjust cross entropy for long-tail cifar experiments
    dy = params['dy']
    ly = params['ly']
    if group_size != 1:
        new_dy = dy.repeat_interleave(group_size)
        new_ly = ly.repeat_interleave(group_size)
        x = logits*F.sigmoid(new_dy)+new_ly
    else:
        x = logits*F.sigmoid(dy)+ly
    if len(params) == 3:
        wy = params[2]
        loss = F.cross_entropy(x, targets, weight=wy)
    else:
        loss = F.cross_entropy(x, targets)
    return loss

def get_trainable_hyper_params(params):
    if isinstance(params,dict):
        return[params[k] for k in params if params[k].requires_grad]
    else:
        return params
def gather_flat_hyper_params(params):
    if isinstance(params,dict):
        return torch.cat([params[k].view(-1) for k in params if params[k].requires_grad])
    else:
        return torch.cat([k.view(-1) for k in params if k.requires_grad])

def assign_hyper_gradient(params, gradient):
    i = 0
    max_len=gradient.shape[0]
    if isinstance(params, dict):
        for k in params:
            para=params[k]
            if para.requires_grad:
                num = para.nelement()
                grad = gradient[i:min(i+num,max_len)].clone()
                torch.reshape(grad, para.shape)
                para.grad = grad.view(para.shape)
                i += num
    else:
        for para in params:
            if para.requires_grad:     
                num = para.nelement()
                grad = gradient[i:min(i+num,max_len)].clone()
                para.grad = grad.view(para.shape)
                i += num