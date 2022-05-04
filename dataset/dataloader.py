import numpy as np
import os.path as osp
from tqdm import tqdm
from dataset.cityscapes import CityscapesDataset
from dataset.gta5 import GTAVDataset
from utils.file_op import load_config

g2c_config = load_config("/media/sdb/duckie/Cross-Domain-Segmentation/configs/gta2cityscapes.yaml")


def create_dataset(mode='G2C'):
    support_mode = ['G2C', 'S2C']
    assert support_mode.__contains__(mode)

    if mode == 'G2C':
        source_dataset = GTAVDataset(root=g2c_config['source_data_dir'],
                                     list_path=osp.join(g2c_config['source_list_dir'], 'train.txt'),
                                     max_iters=g2c_config['max_iter'],
                                     crop_size=tuple(g2c_config['source_input_size']),
                                     mean=tuple(g2c_config['mean']),
                                     std=tuple(g2c_config['std']), set='train',
                                     ignore_label=g2c_config['ignore_label'])

        target_dataset = CityscapesDataset(root=g2c_config['target_data_dir'],
                                           list_path=osp.join(g2c_config['target_list_dir'], 'train.txt'),
                                           max_iters=g2c_config['max_iter'],
                                           crop_size=tuple(g2c_config['target_input_size']),
                                           mean=tuple(g2c_config['mean']),
                                           std=tuple(g2c_config['std']), set='train',
                                           ignore_label=g2c_config['ignore_label'])

        val_dataset = CityscapesDataset(root=g2c_config['target_data_dir'],
                                        list_path=osp.join(g2c_config['target_list_dir'], 'val.txt'),
                                        crop_size=tuple(g2c_config['target_input_size']),
                                        mean=tuple(g2c_config['mean']),
                                        std=tuple(g2c_config['std']), set='val',
                                        ignore_label=g2c_config['ignore_label'])

    return source_dataset, target_dataset, val_dataset


def class_weight(dataloader, num_classes):
    """
    calculate the class weight according to frequency
    """
    result = np.zeros(num_classes)
    print("calculating per-category weight")

    for sample in tqdm(dataloader):
        _, label = sample
        label = np.array(label)
        mask = (label >= 0) & (label < num_classes)
        result += np.bincount(label[mask].astype(np.uint8), minlength=num_classes)

    weights = []
    frequency = []
    num_pixels = np.sum(result)
    for cls_freq in result:
        frequency.append(cls_freq / num_pixels)
        weights.append(1 / (np.log(1.02 + (cls_freq / num_pixels))))

    return np.array(frequency), np.array(weights)
