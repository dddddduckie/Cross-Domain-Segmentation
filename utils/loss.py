import torch
import torch.nn as nn
from utils.lovasz_loss import lovasz_softmax


def ce_loss(pred, target, cuda=True, weight=None,
            batch_average=True, ignore_index=255):
    """
    cross entropy loss
    """
    b, c, h, w = pred.size()
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    if cuda:
        criterion = criterion.cuda()
    loss = criterion(pred, target.long())
    if batch_average:
        loss /= b
    return loss


def bce_loss(pred, target, cuda=True, batch_average=True,
             weight=None):
    """
    binary cross entropy loss
    """
    b, c, h, w = pred.size()
    criterion = nn.BCEWithLogitsLoss(weight=weight)
    if cuda:
        criterion = criterion.cuda()
    loss = criterion(pred, target)
    if batch_average:
        loss /= b
    return loss


def focal_loss(pred, target, cuda=True, weight=None,
               ignore_index=255, gamma=2, alpha=0.5,
               batch_average=True):
    """
    focal loss
    """
    b, c, h, w = pred.size()
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    if cuda:
        criterion = criterion.cuda()

    logpt = -criterion(pred, target.long())
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    if batch_average:
        loss /= b
    return loss


def kl_loss(pred, target, reduction='mean', cuda=True,
            batch_average=True):
    """
    kl-divergence loss
    """
    b, c, h, w = pred.size()
    criterion = nn.KLDivLoss(reduction=reduction)
    if cuda:
        criterion = criterion.cuda()
    loss = criterion(pred, target)
    if batch_average:
        loss /= b
    return loss


def mse_loss(pred, target, cuda=True, batch_average=True):
    """
    mean squared error loss
    """
    b, c, h, w = pred.size()
    criterion = torch.nn.MSELoss()
    if cuda:
        criterion = criterion.cuda()
    loss = criterion(pred, target)
    if batch_average:
        loss /= b
    return loss


def lovasz_loss(pred, target, cuda=True, batch_average=True, ignore_index=255):
    """
    lovasz softmax loss
    """
    b, c, h, w = pred.size()
    criterion = lovasz_softmax(probas=pred, labels=target, ignore=ignore_index)
    if cuda:
        criterion = criterion.cuda()
    loss = criterion
    if batch_average:
        loss /= b
    return loss


def compute_loss(pred, target, name=None, cuda=True, weight=None, ignore_index=255):
    support_loss = ['ce', 'bce', 'fl', 'kl', 'mse', 'lovasz']
    assert support_loss.__contains__(name)
    if name == "ce":
        return ce_loss(pred=pred, target=target, weight=weight,
                       cuda=cuda, ignore_index=ignore_index)
    elif name == "bce":
        return bce_loss(pred=pred, target=target, weight=weight, cuda=cuda)
    elif name == "fl":
        return focal_loss(pred=pred, target=target, weight=weight,
                          cuda=cuda, ignore_index=ignore_index)
    elif name == "kl":
        return kl_loss(pred=pred, target=target, cuda=cuda)
    elif name == "mse":
        return mse_loss(pred=pred, target=target, cuda=cuda)
    elif name == "lovasz":
        return lovasz_loss(pred=pred, target=target, ignore_index=ignore_index, cuda=cuda)
