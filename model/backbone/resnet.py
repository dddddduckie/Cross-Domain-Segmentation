import torch
import torch.nn as nn
from utils.net_util import set_require_grad, weight_initialization
from model.backbone.layer_factory import BottleNeck, ConvLayer

model_path = "/media/sdb/duckie/Cross-Domain-Segmentation/model/pretrained_models/resnet101-5d3b4d8f.pth"


class ResNet(nn.Module):
    """
    resnet backbone
    """

    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.in_dim = 64
        self.layer1 = []
        planes = [64, 128, 256, 512]

        # stage 1
        self.layer1 += [ConvLayer(in_dim=3, out_dim=64, kernel_size=7, stride=2,
                                  padding=3, norm='bn', activation='relu', use_bias=False)]
        self.layer1 += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)]
        self.layer1 = nn.Sequential(*self.layer1)

        # stage 2-5
        self.layer2 = self._make_layer(block, planes[0], layers[0])
        self.layer3 = self._make_layer(block, planes[1], layers[1], stride=2)
        self.layer4 = self._make_layer(block, planes[2], layers[2], stride=1, dilation=(2, 2))
        self.layer5 = self._make_layer(block, planes[3], layers[3], stride=1, dilation=(4, 4))

    def _make_layer(self, block, dim, num_blocks, stride=1, dilation=(1, 1)):
        downsample = None
        if stride != 1 or self.in_dim != dim * block.expansion or dilation[0] == 2 or dilation[0] == 4:
            downsample = nn.Sequential(ConvLayer(in_dim=self.in_dim, out_dim=dim * block.expansion, kernel_size=1,
                                                 stride=stride, norm='bn', use_bias=False))
            set_require_grad(downsample._modules['0'].norm, False)
        layers = []
        layers += [block(in_dim=self.in_dim, out_dim=dim, stride=stride, downsample=downsample,
                         dilation=dilation)]
        self.in_dim = dim * block.expansion
        for i in range(num_blocks - 1):
            layers += [block(in_dim=self.in_dim, out_dim=dim, dilation=dilation)]

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out0 = out
        out = self.layer5(out)

        return out0, out


def load_pretrained_resnet(path, model):
    """
    load a pretrained backbone
    """
    pretrained_dict = torch.load(path)
    pretrained_values = []
    new_values = []

    for i in pretrained_dict.values():
        pretrained_values.append(i)

    j = 0
    while j < len(pretrained_values) - 2:
        new_values.append(pretrained_values[j])

        # comment to use instance normalization
        new_values.append(pretrained_values[j + 3])
        new_values.append(pretrained_values[j + 4])
        new_values.append(pretrained_values[j + 1])
        new_values.append(pretrained_values[j + 2])
        j = j + 5

    new_state_dict = model.state_dict()
    cur = 0

    for k in new_state_dict:
        if k.split('.')[-1] == 'num_batches_tracked':
            continue
        else:
            new_state_dict[k] = new_values[cur]
            cur = cur + 1
    model.load_state_dict(new_state_dict)


def resnet_101(pre_trained=True):
    resnet = ResNet(BottleNeck, [3, 4, 23, 3])
    if pre_trained:
        load_pretrained_resnet(model_path, resnet)
    else:
        resnet = weight_initialization(resnet)
    return resnet


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_101 = resnet_101()
    resnet_101 = resnet_101.to(device)
    image = torch.ones(1, 3, 700, 1280)
    image = image.to(device)
    output = resnet_101(image)
    print(output)
    print(output.size())
