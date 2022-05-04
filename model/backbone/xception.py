import torch.nn as nn
import torch
from model.backbone.layer_factory import XceptionBlock, ConvLayer
from utils.net_util import weight_initialization


class AlignedXception(nn.Module):
    """
    Modified Aligned Xception
    """

    def __init__(self, output_stride):
        super(AlignedXception, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError

        self.entry_flow = []
        self.middle_flow = []
        self.exit_flow = []

        # construct entry flow
        self.entry_flow += [ConvLayer(in_dim=3, out_dim=32, kernel_size=3, stride=2,
                                      padding=1, use_bias=False, activation='relu',
                                      norm='bn')]
        self.entry_flow += [ConvLayer(in_dim=32, out_dim=64, kernel_size=3, stride=1,
                                      padding=1, use_bias=False, activation='relu',
                                      norm='bn')]
        self.entry_flow += [XceptionBlock(64, 128, reps=2, stride=2, start_with_relu=False)]
        self.entry_flow += [nn.ReLU(inplace=True)]
        self.entry_flow += [XceptionBlock(128, 256, reps=2, stride=2, start_with_relu=False,
                                          grow_first=True)]
        self.entry_flow += [XceptionBlock(256, 728, reps=2, stride=entry_block3_stride,
                                          start_with_relu=True, grow_first=True, is_last=True)]

        # construct middle flow
        for i in range(16):
            self.middle_flow += [XceptionBlock(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                                               start_with_relu=True, grow_first=True)]

        # construct exit flow
        self.exit_flow += [XceptionBlock(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                                         start_with_relu=True, grow_first=False, is_last=True)]
        self.exit_flow += [nn.ReLU(inplace=True)]
        self.exit_flow += [ConvLayer(1024, 1024, kernel_size=3, dilation=exit_block_dilations[1], norm='bn',
                                     groups=1024, use_bias=False, fixed_padding=True)]
        self.exit_flow += [ConvLayer(1024, 1536, kernel_size=1, use_bias=False, norm='bn', activation='relu')]
        self.exit_flow += [ConvLayer(1536, 1536, kernel_size=3, dilation=exit_block_dilations[1], norm='bn',
                                     groups=1536, use_bias=False, fixed_padding=True)]
        self.exit_flow += [ConvLayer(1536, 1536, kernel_size=1, use_bias=False, norm='bn', activation='relu')]
        self.exit_flow += [ConvLayer(1536, 1536, kernel_size=3, dilation=exit_block_dilations[1], norm='bn',
                                     groups=1536, use_bias=False, fixed_padding=True)]
        self.exit_flow += [ConvLayer(1536, 2048, kernel_size=1, use_bias=False, norm='bn', activation='relu')]

        self.entry_flow = nn.Sequential(*self.entry_flow)
        self.middle_flow = nn.Sequential(*self.middle_flow)
        self.exit_flow = nn.Sequential(*self.exit_flow)

    def forward(self, x):
        entry_feature = self.entry_flow(x)
        middle_feature = self.middle_flow(entry_feature)
        exit_feature = self.exit_flow(middle_feature)

        return entry_feature, middle_feature, exit_feature


def modified_xception(pre_trained=True):
    xception = AlignedXception(output_stride=16)
    if pre_trained:
        load_pretrained_xception("../pretrained_models/xception.pth", xception)
    else:
        xception = weight_initialization(xception)
    return xception


def load_pretrained_xception(path, model):
    """
    load a pretrained backbone
    """
    pretrained_dict = torch.load(path)
    pretrained_values = []

    for i in pretrained_dict.values():
        pretrained_values.append(i)
    new_state_dict = model.state_dict()

    cur = 0
    for k in new_state_dict:
        new_state_dict[k] = pretrained_values[cur]
        cur = cur + 1

    model.load_state_dict(new_state_dict)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xception = modified_xception(pre_trained=True)
    xception.to(device)
    image = torch.ones(1, 3, 700, 1280)
    image = image.to(device)
    _, _, exit_feature = xception(image)
    print(exit_feature)
