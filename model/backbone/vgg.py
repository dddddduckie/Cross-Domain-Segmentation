import torch.nn as nn
import torch
from model.backbone.layer_factory import ConvLayer
from utils.net_util import weight_initialization


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg = []
        self.vgg += [ConvLayer(in_dim=3, out_dim=64, kernel_size=3, padding=1,
                               activation='relu', norm='bn')]
        self.vgg += [ConvLayer(in_dim=64, out_dim=64, kernel_size=3, padding=1,
                               activation='relu', norm='bn')]
        self.vgg += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.vgg += [ConvLayer(in_dim=64, out_dim=128, kernel_size=3, padding=1,
                               activation='relu', norm='bn')]
        self.vgg += [ConvLayer(in_dim=128, out_dim=128, kernel_size=3, padding=1,
                               activation='relu', norm='bn')]
        self.vgg += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.vgg += [ConvLayer(in_dim=128, out_dim=256, kernel_size=3, padding=1,
                               activation='relu', norm='bn')]
        for i in range(2):
            self.vgg += [ConvLayer(in_dim=256, out_dim=256, kernel_size=3, padding=1,
                                   activation='relu', norm='bn')]
        self.vgg += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.vgg += [ConvLayer(in_dim=256, out_dim=512, kernel_size=3, padding=1,
                               activation='relu', norm='bn')]
        for i in range(7):
            if i == 2 or i == 6:
                self.vgg += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                self.vgg += [ConvLayer(in_dim=512, out_dim=512, kernel_size=3, padding=1,
                                       activation='relu', norm='bn')]
        self.vgg = nn.Sequential(*self.vgg)

    def forward(self, x):
        output = self.vgg(x)
        return output


def vgg_16(pre_trained=True):
    vgg = VGG()
    if pre_trained:
        load_pretrained_vgg("../pretrained_models/vgg16_bn-6c64b313.pth", vgg)
    else:
        vgg = weight_initialization(vgg)
    return vgg


def load_pretrained_vgg(path, model):
    """
    load a pretrained backbone
    """
    pretrained_dict = torch.load(path)
    pretrained_values = []

    for i in pretrained_dict.values():
        pretrained_values.append(i)
    pretrained_values = pretrained_values[:len(pretrained_values) - 6]

    new_state_dict = model.state_dict()

    cur = 0
    for k in new_state_dict:
        if k.split('.')[-1] == 'num_batches_tracked':
            continue
        else:
            new_state_dict[k] = pretrained_values[cur]
            cur = cur + 1

    model.load_state_dict(new_state_dict)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg16 = vgg_16(pre_trained=True)
    image = torch.ones(1, 3, 700, 1280)
    image = image.to(device)
    feature = vgg16(image)
    print(feature)
