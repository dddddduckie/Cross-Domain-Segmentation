import os
import shutil
import yaml
import torch
import os.path as osp
from tensorboardX import SummaryWriter


def mkdir(path):
    """
    make directory if not existed
    """
    if not osp.exists(path):
        os.makedirs(path)


def load_config(file):
    """
    load configuration file
    """
    with open(file, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def sava_checkpoint(state_dict, path, prefix):
    """
    save model state dict to the given path
    """
    mkdir(path)
    filename = osp.join(path, '{}_{}.pth.tar'.
                        format(prefix, state_dict['iter']))
    torch.save(state_dict, filename)


def add_summary(scalar, iter, dir, freq=10):
    """
    write summary
    """
    writer = SummaryWriter(logdir=dir)
    if iter % freq == 0:
        for key, val in scalar.items():
            writer.add_scalar(key, val, iter)
    writer.close()


def merge_file(src, dst):
    """
    Merge several directories into one.
    """
    dir_list = os.listdir(src)
    for dir in dir_list:
        child = os.path.join(src, dir)
        if os.path.isfile(child):
            shutil.copy(child, dst)
            continue
        merge_file(child, dst)
