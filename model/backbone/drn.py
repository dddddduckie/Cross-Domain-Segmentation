import torch.nn as nn
import torch
from model.backbone.layer_factory import BottleNeck, ConvLayer
from utils.net_util import weight_initialization


class DRN(nn.Module):
    def __init__(self, block, layers,
                 planes=(16, 32, 64, 128, 256, 512, 512, 512)):
        super(DRN, self).__init__()
        self.in_dim = planes[0]
        self.out_dim = planes[-1]
        self.layer0 = []
        self.layer0 += [ConvLayer(in_dim=3, out_dim=planes[0], kernel_size=7, padding=3,
                                  use_bias=False, norm='bn', activation='relu')]
        self.layer0 = nn.Sequential(*self.layer0)
        self.layer1 = self._make_conv_layers(planes[0], layers[0], stride=1)
        self.layer2 = self._make_conv_layers(planes[1], layers[1], stride=2)

        self.layer3 = self._make_layers(block, planes[2], layers[2], stride=2)
        self.layer4 = self._make_layers(block, planes[3], layers[3], stride=2)
        self.layer5 = self._make_layers(block, planes[4], layers[4], dilation=2,
                                        new_level=False)
        self.layer6 = self._make_layers(block, planes[5], layers[5], dilation=4,
                                        new_level=False)

        self.layer7 = self._make_conv_layers(planes[6], layers[6], dilation=2)
        self.layer8 = self._make_conv_layers(planes[7], layers[7], dilation=1)

    def _make_conv_layers(self, dim, convs, stride=1, dilation=1):
        layers = []
        for i in range(convs):
            layers += [ConvLayer(in_dim=self.in_dim, out_dim=dim, kernel_size=3,
                                 stride=stride if i == 0 else 1,
                                 padding=dilation, use_bias=False, dilation=dilation,
                                 norm='bn', activation='relu')]
            self.in_dim = dim
        return nn.Sequential(*layers)

    def _make_layers(self, block, dim, num_blocks, stride=1, dilation=1,
                     new_level=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.in_dim != dim * block.expansion:
            downsample = nn.Sequential(ConvLayer(in_dim=self.in_dim, out_dim=dim * block.expansion, kernel_size=1,
                                                 stride=stride, norm='bn', use_bias=False))
            # set_require_grad(downsample._modules['0'].norm, False)
        layers = []

        layers += [block(in_dim=self.in_dim, out_dim=dim, stride=stride, downsample=downsample,
                         dilation=(1, 1) if dilation == 1 else (
                             dilation // 2 if new_level else dilation, dilation)
                         )]

        self.in_dim = dim * block.expansion
        for i in range(num_blocks - 1):
            layers += [block(in_dim=self.in_dim, out_dim=dim, dilation=(dilation, dilation))]

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        low_level_feature = out
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        return out, low_level_feature


def drn_d_105(pre_trained=True):
    """
    drn_d_105
    """
    drn = DRN(BottleNeck, [1, 1, 3, 4, 23, 3, 1, 1])
    if pre_trained:
        load_pretrained_drn("../pretrained_models/drn_d_105-12b40979.pth", drn)
    else:
        drn = weight_initialization(drn)
    return drn


def load_pretrained_drn(path, model):
    """
    load a pretrained backbone
    """
    pretrained_dict = torch.load(path)
    del pretrained_dict['fc.weight']
    del pretrained_dict['fc.bias']

    pretrained_values = []

    for i in pretrained_dict.values():
        pretrained_values.append(i)

    new_state_dict = model.state_dict()
    cur = 0

    for k in new_state_dict:
        if k.split('.')[-1] == 'num_batches_tracked':
            continue
        else:
            new_state_dict[k] = pretrained_values[cur]
            cur = cur + 1
    model.load_state_dict(new_state_dict)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drn = drn_d_105(pretrained=True)
    drn = drn.to(device)
    image = torch.ones(1, 3, 512, 512)
    image = image.to(device)
    output, low_level_feature = drn(image)
    print(output)
