from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    """
    base dataset class
    """

    def __init__(self, root, list_path, max_iters, crop_size, mean, std, set):
        self.root = root
        self.list_path = list_path
        self.max_iters = max_iters
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.set = set
        with open(self.list_path) as f:
            self.img_ids = [i_id.strip() for i_id in f]
        # use epoch instead
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.data = self.load_image_and_label()

    def load_image_and_label(self):
        raise NotImplementedError

    def augment(self, example):
        raise NotImplementedError

    def __len__(self):
        return 0

    def __getitem__(self, item):
        pass
