import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
import torch.optim as opt
import torch


def set_require_grad(module, value):
    """
    set gradient requirement
    """
    for i in module.parameters():
        i.requires_grad = value


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


def l1_norm(model, loss):
    """
    add l1 regularization to the current loss
    """
    for param in model.parameters():
        loss += torch.sum(torch.abs(param))
    # loss.backward()
    return loss


def adjust_learning_rate(optimizer, cur_iter, ini_lr, step_size,
                         max_iter, max_lr=None, mode='fixed'):
    """
    adjust the learning rate of optimizer
    """
    support_mode = ['fixed', 'exp', 'step', 'cos', 'poly', 'cyclical']
    assert support_mode.__contains__(mode), "unsupported learning rate policy"

    if mode == 'fixed':
        lr = ini_lr
    # exponential mode decays fast
    elif mode == 'exp':
        lr = ini_lr * pow(0.9999, cur_iter)
    elif mode == 'step':
        lr = ini_lr * pow(0.1, cur_iter // step_size)
    elif mode == 'cos':
        lr = ini_lr * 0.5 * (1 + math.cos(1.0 * cur_iter / max_iter * math.pi))
    elif mode == 'poly':
        lr = ini_lr * pow((1 - 1.0 * cur_iter / max_iter), 0.9)
    elif mode == 'cyclical':
        cycle = np.floor(1 + cur_iter / (2 * step_size))
        x = np.abs(cur_iter / step_size - 2 * cycle + 1)
        lr = ini_lr + (max_lr - ini_lr) * np.maximum(0, (1 - x))

    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        for i in range(1, len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr * 10


def weight_initialization(net, init_type='kaiming'):
    support_init_type = ['gaussian', 'xavier', 'kaiming', 'orthogonal']
    assert support_init_type.__contains__(init_type), "unsupported initialization type"
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            if init_type == 'gaussian':
                nn.init.normal_(module.weight.data, 0.0, 0.01)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(module.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(module.weight.data, gain=math.sqrt(2))
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.01)
            nn.init.zeros_(module.bias)
    return net


def create_optimizer(params, lr, momentum=None, weight_decay=None,
                     betas=None, name="SGD"):
    support_optimizer = ['SGD', 'Adagrad', 'Adadelta', 'Adam', 'LBFGS', 'RMSprop']
    assert support_optimizer.__contains__(name), "unsupported optimizer"
    if name == "SGD":
        optimizer = opt.SGD(params=params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "Adagrad":
        optimizer = opt.Adagrad(params=params, lr=lr)
    elif name == "Adadelta":
        optimizer = opt.Adadelta(params=params, lr=lr)
    elif name == "Adam":
        optimizer = opt.Adam(params=params, lr=lr, betas=betas)
    elif name == "LBFGS":
        optimizer = opt.LBFGS(params=params, lr=lr)
    elif name == "RMSprop":
        optimizer = opt.RMSprop(params=params, lr=lr)

    return optimizer
