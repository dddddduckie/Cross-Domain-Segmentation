import numpy as np


def confusion_matrix(pred, gt, num_classes):
    """
    compute the confusion matrix
    """
    mask = (gt >= 0) & (gt < num_classes)
    label = num_classes * gt[mask].astype('int') + pred[mask]
    count = np.bincount(label, minlength=num_classes ** 2)
    confusion_matrix = count.reshape(num_classes, num_classes)
    return confusion_matrix


def per_cls_iou(confusion_matrix):
    """
    compute per-class iou
    """
    iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) +
                                       confusion_matrix.sum(1) - np.diag(confusion_matrix))
    return iou


def per_cls_acc(confusion_matrix):
    """
    compute per-class pixel-level accuracy
    """
    acc = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    return acc


def mIoU(confusion_matrix):
    """
    compute mIoU
    """
    per_class_iou = per_cls_iou(confusion_matrix)
    return np.nanmean(per_class_iou)


def acc(confusion_matrix):
    """
    compute overall pixel-level accuracy
    """
    per_class_acc = per_cls_acc(confusion_matrix)
    return np.nanmean(per_class_acc)
