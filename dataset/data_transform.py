from __future__ import print_function, division

import torch
import numpy as np
import random
from PIL import Image, ImageFilter


class Normalize(object):
    """
    Normalize the given tensor image
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, example):
        image = np.asarray(example['image']).astype(np.float32)
        label = np.asarray(example['label']).astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std

        return {'image': image, 'label': label}


class ToTensor(object):
    """
    convert numpy image (H×W×C) to tensor (C×H×W)
    """

    def __call__(self, example):
        image = np.asarray(example['image'], np.float32).transpose((2, 0, 1))
        label = np.asarray(example['label'], np.float32)

        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(label).float()}


class CentralCrop(object):
    """
    Crop the image centrally
    """

    def __init__(self, crop_size_h, crop_size_w):
        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w

    def __call__(self, example):
        image, label = example['image'], example['label']
        w, h = image.size
        assert 0 < self.crop_size_h < h
        assert 0 < self.crop_size_w < w

        h_start = (h - self.crop_size_h) // 2
        w_start = (w - self.crop_size_w) // 2
        image = image.crop((w_start, h_start, w_start + self.crop_size_w, h_start + self.crop_size_h))
        label = label.crop((w_start, h_start, w_start + self.crop_size_w, h_start + self.crop_size_h))

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop the image randomly
    """

    def __init__(self, crop_size_h, crop_size_w):
        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w

    def __call__(self, example):
        image, label = example['image'], example['label']
        w, h = image.size
        assert 0 < self.crop_size_h < h
        assert 0 < self.crop_size_w < w

        h_start = np.random.randint(0, h - self.crop_size_h + 1)
        w_start = np.random.randint(0, w - self.crop_size_w + 1)
        image = image.crop((w_start, h_start, w_start + self.crop_size_w, h_start + self.crop_size_h))
        label = label.crop((w_start, h_start, w_start + self.crop_size_w, h_start + self.crop_size_h))

        return {'image': image, 'label': label}


class RandomFlip(object):
    """
    Flip the image randomly according to the direction
    """

    def __init__(self, mode="horizon"):
        self.mode = mode

    def __call__(self, example):
        image, label = example['image'], example['label']
        rand = random.random()
        if rand > 0.5:
            if self.mode == "horizon":
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            elif self.mode == "vertical":
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                label = label.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': image, 'label': label}


class Resize(object):
    """
    Resize the image to a given size
    """

    def __init__(self, h, w):
        self.height = h
        self.width = w

    def __call__(self, example):
        image, label = example['image'], example['label']

        image = image.resize((self.width, self.height), Image.BICUBIC)
        label = label.resize((self.width, self.height), Image.NEAREST)

        return {'image': image, 'label': label}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, example):
        image, label = example['image'], example['label']
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        image = image.rotate(rotate_degree, Image.BICUBIC)
        label = label.rotate(rotate_degree, Image.NEAREST)

        return {'image': image, 'label': label}


class RandomGaussianBlur(object):
    def __call__(self, example):
        image, label = example['image'], example['label']
        rand = random.random()
        if rand < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=rand))

        return {'image': image, 'label': label}


class Compose(object):
    """
    compose several data transformation
    """

    def __init__(self, transformation):
        self.transformation = transformation

    def __call__(self, example):
        result = example
        for tran in self.transformation:
            result = tran(result)
        return result
