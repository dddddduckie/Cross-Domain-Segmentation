import torch.nn as nn
import torch
from model.backbone.layer_factory import ConvLayer, InvertedResidual
from utils.net_util import weight_initialization

model_path="/media/sdb/duckie/Cross-Domain-Segmentation/model/pretrained_models/mobilenet_v2-6a65762b.pth"
# note the original version uses RELU6

class MobileNetV2(nn.Module):
    """
    mobilenet v2
    """

    def __init__(self, block=None, width_mult=1., output_stride=8):
        super(MobileNetV2, self).__init__()
        self.block = block
        in_dim = 32
        cur_stride = 1
        rate = 1
        self.mobilenet = []

        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        in_dim = int(in_dim * width_mult)

        # build the first layer
        self.mobilenet += [ConvLayer(in_dim=3, out_dim=in_dim, kernel_size=3, stride=2, norm='bn',
                                     padding=1, use_bias=False, activation='relu')]
        cur_stride *= 2

        for t, c, n, s in interverted_residual_setting:
            if cur_stride == output_stride:
                stride = 1
                dilation = rate
                rate *= s
            else:
                stride = s
                dilation = 1
                cur_stride *= s
            out_dim = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.mobilenet += [self.block(in_dim=in_dim, out_dim=out_dim, stride=stride,
                                                  dilation=dilation, expand_ratio=t)]
                else:
                    self.mobilenet += [self.block(in_dim=in_dim, out_dim=out_dim, stride=1,
                                                  dilation=dilation, expand_ratio=t)]
                in_dim = out_dim

        self.mobilenet = nn.Sequential(*self.mobilenet)
        self.low_level_feature = self.mobilenet[0:4]
        self.high_level_feature = self.mobilenet[4:]

    def forward(self, x):

        low_level_feature = self.low_level_feature(x)
        high_level_feature = self.high_level_feature(low_level_feature)
        return low_level_feature, high_level_feature


def mobilenet_v2(pre_trained=True):
    mbnet = MobileNetV2(block=InvertedResidual, output_stride=16)
    if pre_trained:
        load_pretrained_mobilenet(model_path, mbnet)
    else:
        mbnet = weight_initialization(mbnet)
    return mbnet


def load_pretrained_mobilenet(path, model):
    """
    load a pretrained backbone
    """
    pretrained_dict = torch.load(path)
    pretrained_values = []

    for i in pretrained_dict.values():
        pretrained_values.append(i)

    new_values = pretrained_values[0:len(pretrained_values) - 5]
    count = len(new_values)

    new_state_dict = model.state_dict()
    cur = 0

    for k in new_state_dict:
        if cur == count:
            break
        if k.split('.')[-1] == 'num_batches_tracked':
            continue
        else:
            new_state_dict[k] = new_values[cur]
            cur = cur + 1
    model.load_state_dict(new_state_dict)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mobilenet = mobilenet_v2()
    mobilenet = mobilenet.to(device)
    image = torch.ones(1, 3, 700, 1280)
    image = image.to(device)
    low_level_feature, high_level_feature = mobilenet(image)
    print(high_level_feature)
