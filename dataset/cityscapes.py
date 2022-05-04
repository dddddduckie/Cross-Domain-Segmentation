from dataset.base_dataset import BaseDataset
import os.path as osp
from dataset.data_transform import *
from torch.utils import data
import matplotlib.pyplot as plt
from utils.data_visualization import visualize_segmap


class CityscapesDataset(BaseDataset):
    """
    The cityscapes dataset
    """

    def __init__(self, root, list_path, max_iters=None, crop_size=(512, 1024),
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), ignore_label=255, set='val'):
        super().__init__(root, list_path, max_iters, crop_size, mean, std, set)
        self.ignore_label = ignore_label
        self.id_to_trainid = {
            0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2,
            12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
            24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: -1
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        datafiles = self.data[item]
        example = {}
        example["image"] = Image.open(datafiles["image"]).convert('RGB')
        example["label"] = Image.open(datafiles["label"])
        result = self.augment(example)
        result["label"] = self.convert_label(result["label"])
        result["name"] = datafiles["name"]
        return result["image"], result["label"], result["name"]

    def convert_label(self, label):
        train_label = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            train_label[label == k] = v
        return train_label

    def augment(self, example):
        composed_transform = Compose([
            Resize(self.crop_size[0], self.crop_size[1]),
            Normalize(mean=self.mean, std=self.std),
            ToTensor()
        ])
        return composed_transform(example)

    def load_image_and_label(self):
        file = []
        for id in self.img_ids:
            image = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, id))
            id_split = id.split('_')
            label = osp.join(self.root, "gtFine/%s/%s" % (self.set, id_split[0] + '_' + id_split[1]
                                                          + '_' + id_split[2] + '_' + 'gtFine_labelIds.png'))
            file.append({
                "image": image,
                "label": label,
                "name": id
            })
        return file


if __name__ == '__main__':
    cityscapes_dataset = CityscapesDataset('/media/sdb/duckie/dataset/Cityscapes',
                                           list_path='./cityscapes_list/val.txt')
    trainloader = data.DataLoader(cityscapes_dataset, batch_size=4, shuffle=True)

    for i, data in enumerate(trainloader):
        imgs, labels, _ = data
        if i == 0:
            img_0 = imgs[0].numpy()
            label_0 = labels[0].numpy()
            img_0 = img_0.transpose(1, 2, 0)
            corlor_map = visualize_segmap(label_0, dataset="cityscapes")

            img_0 *= (0.229, 0.224, 0.225)
            img_0 += (0.485, 0.456, 0.406)
            img_0 *= 255.0
            img_0 = img_0.astype(np.uint8)

            plt.figure()
            plt.title("data visualization")
            plt.subplot(211)
            plt.imshow(img_0)
            plt.subplot(212)
            plt.imshow(corlor_map)
            plt.show()
            break
