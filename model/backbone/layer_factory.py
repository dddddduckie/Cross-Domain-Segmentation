import torch.nn.functional as F

from torch import nn
from utils.net_util import set_require_grad, fixed_padding


class LinearLayer(nn.Module):
    """
    Fully connected layers.
    """

    def __init__(self, in_dim, out_dim, activation='none', norm='none', use_bias=True,
                 dropout='none'):
        super(LinearLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)
        if dropout != 'none':
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        support_act = ['relu', 'prelu', 'lrelu', 'selu', 'tanh', 'sigmoid', 'none']
        support_norm = ['bn', 'in', 'none']
        assert support_act.__contains__(activation)
        assert support_norm.__contains__(norm)

        # set activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None

        # set normalization function
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(out_dim)
        elif norm == 'none':
            self.norm = None

    def forward(self, x):
        out = self.fc(x)
        if self.dropout:
            out = self.dropout(out)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class ConvLayer(nn.Module):
    """
    Basic convolutional block
    """

    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0,
                 fixed_padding=False, dilation=1, activation='none', norm='none',
                 use_bias=True, groups=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, bias=use_bias,
                              padding=padding, dilation=dilation, groups=groups)

        support_act = ['relu', 'prelu', 'lrelu', 'selu', 'tanh', 'sigmoid', 'none']
        support_norm = ['bn', 'gn', 'in', 'none']
        assert support_act.__contains__(activation)
        assert support_norm.__contains__(norm)
        self.fixed_padding = fixed_padding

        # set activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None

        # set normalization function
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_dim)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(num_groups=2, num_channels=out_dim)
        elif norm == 'none':
            self.norm = None

    def forward(self, x):
        if self.fixed_padding:
            x = fixed_padding(x, self.conv.kernel_size[0], dilation=self.conv.dilation[0])
        out = self.conv(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class SE_Layer(nn.Module):
    """
    squeeze and excitation layer
    """

    def __init__(self, dim, reduction=16):
        super(SE_Layer, self).__init__()
        self.se = []
        self.se += [nn.AdaptiveAvgPool2d(1)]
        self.se += [ConvLayer(in_dim=dim, out_dim=dim // reduction, kernel_size=1,
                              activation='relu', use_bias=False)]
        self.se += [ConvLayer(in_dim=dim // reduction, out_dim=dim, kernel_size=1,
                              activation='sigmoid', use_bias=False)]
        self.se = nn.Sequential(*self.se)

    def forward(self, x):
        out = self.se(x)
        return out * x


class MLP(nn.Module):
    """
    multi layer perceptron
    """

    def __init__(self, in_dim, mid_dim, out_dim, num_blocks, norm='none', activation='relu',
                 dropout='none'):
        super(MLP, self).__init__()
        self.mlp = []
        self.mlp += [LinearLayer(in_dim=in_dim, out_dim=mid_dim, norm=norm, activation=activation,
                                 dropout=dropout)]
        for i in range(num_blocks - 2):
            self.mlp += [LinearLayer(in_dim=mid_dim, out_dim=mid_dim, norm=norm, activation=activation,
                                     dropout=dropout)]
        self.mlp += [LinearLayer(in_dim=mid_dim, out_dim=out_dim, norm=norm, activation='none',
                                 dropout=dropout)]
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        out = self.mlp(x.view(x.size(0), -1))
        return out


class BasicBlcok(nn.Module):
    """
    basic block w/o bottleneck
    """
    expansion = 1

    def __init__(self, in_dim, out_dim, activation='relu', stride=1,
                 dilation=1, norm='bn', se=False, downsample=None):
        super(BasicBlcok, self).__init__()
        self.block = []
        self.block += [ConvLayer(in_dim=in_dim, out_dim=out_dim, norm=norm, activation=activation,
                                 kernel_size=3, stride=stride, padding=1, use_bias=False, dilation=dilation)]
        self.block += [ConvLayer(in_dim=out_dim, out_dim=out_dim, norm=norm, activation='none',
                                 kernel_size=3, stride=1, padding=1, use_bias=False, dilation=dilation)]
        self.block = nn.Sequential(*self.block)
        self.downsample = downsample
        self.stride = stride

        # squeeze and excitation
        if se:
            self.se = SE_Layer(out_dim)
        else:
            self.se = None

    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return F.relu(out, inplace=True)


class BottleNeck(nn.Module):
    """
    bottleneck structure
    """
    expansion = 4

    def __init__(self, in_dim, out_dim, activation='relu', stride=1,
                 dilation=(1, 1), norm='bn', se=False, downsample=None):
        super(BottleNeck, self).__init__()
        padding = dilation[1]
        self.bottleneck = []
        self.bottleneck += [ConvLayer(in_dim=in_dim, out_dim=out_dim, kernel_size=1, stride=1,
                                      activation=activation, norm=norm, use_bias=False)]
        self.bottleneck += [ConvLayer(in_dim=out_dim, out_dim=out_dim, kernel_size=3, stride=stride,
                                      activation=activation, norm=norm, use_bias=False, padding=padding,
                                      dilation=dilation[1])]
        self.bottleneck += [ConvLayer(in_dim=out_dim, out_dim=out_dim * 4, kernel_size=1, stride=1,
                                      activation='none', norm=norm, use_bias=False)]
        self.bottleneck = nn.Sequential(*self.bottleneck)
        self.downsample = downsample
        self.stride = stride

        # squeeze and excitation
        if se:
            self.se = SE_Layer(out_dim * 4)
        else:
            self.se = None

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return F.relu(out, inplace=True)


class ASPP_V2(nn.Module):
    """
    atrous spatial pyramid pooling layer
    """

    def __init__(self, in_dim, dilation_rates, padding_rates, num_classes):
        super(ASPP_V2, self).__init__()
        self.conv_list = nn.ModuleList()
        for dilation, padding in zip(dilation_rates, padding_rates):
            self.conv_list.append(ConvLayer(in_dim=in_dim, out_dim=num_classes, kernel_size=3, stride=1,
                                            padding=padding, dilation=dilation))

    def forward(self, x):
        out = self.conv_list[0](x)
        for i in range(len(self.conv_list) - 1):
            out += self.conv_list[i + 1](x)
        return out


class InvertedResidual(nn.Module):
    """
    inverted residual block
    """

    def __init__(self, in_dim, out_dim, stride, dilation, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.dilation = dilation
        self.inverted_residual = []
        mid_dim = round(in_dim * expand_ratio)
        self.residual = self.stride == 1 and in_dim == out_dim
        if expand_ratio != 1:
            self.inverted_residual += [ConvLayer(in_dim, mid_dim, kernel_size=1, use_bias=False,
                                                 activation='relu', norm='bn')]
        # depth-wise convolution
        self.inverted_residual += [ConvLayer(mid_dim, mid_dim, kernel_size=3, dilation=dilation,
                                             groups=mid_dim, stride=stride, use_bias=False,
                                             activation='relu', norm='bn')]
        self.inverted_residual += [ConvLayer(mid_dim, out_dim, kernel_size=1, use_bias=False, activation='none',
                                             norm='bn')]
        self.inverted_residual = nn.Sequential(*self.inverted_residual)

    def forward(self, x):
        x_pad = fixed_padding(x, kernel_size=3, dilation=self.dilation)
        if self.residual:
            result = self.inverted_residual(x_pad) + x
        else:
            result = self.inverted_residual(x_pad)
        return result


class XceptionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, reps, stride=1, dilation=1,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(XceptionBlock, self).__init__()

        if in_dim != out_dim or stride != 1:
            self.downsample = ConvLayer(in_dim, out_dim, kernel_size=1, stride=stride, use_bias=False,
                                        norm='bn')
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=True)
        self.xception_block = []

        filters = in_dim
        if grow_first:
            self.xception_block.append(self.relu)
            self.xception_block.append(ConvLayer(in_dim, in_dim, kernel_size=3, dilation=dilation, norm='bn',
                                                 groups=in_dim, use_bias=False, fixed_padding=True))
            self.xception_block.append(ConvLayer(in_dim, out_dim, kernel_size=1, use_bias=False, norm='bn'))
            filters = out_dim

        for i in range(reps - 1):
            self.xception_block.append(self.relu)
            self.xception_block.append(ConvLayer(filters, filters, kernel_size=3, dilation=dilation, norm='bn',
                                                 groups=filters, use_bias=False, fixed_padding=True))
            self.xception_block.append(ConvLayer(filters, filters, kernel_size=1, use_bias=False, norm='bn'))

        if not grow_first:
            self.xception_block.append(self.relu)
            self.xception_block.append(ConvLayer(in_dim, in_dim, kernel_size=3, dilation=dilation, norm='bn',
                                                 groups=in_dim, use_bias=False, fixed_padding=True))
            self.xception_block.append(ConvLayer(in_dim, out_dim, kernel_size=1, use_bias=False, norm='bn'))

        if stride != 1:
            self.xception_block.append(self.relu)
            self.xception_block.append(ConvLayer(out_dim, out_dim, kernel_size=3, stride=stride, norm='bn',
                                                 groups=out_dim, use_bias=False, fixed_padding=True))
            self.xception_block.append(ConvLayer(out_dim, out_dim, kernel_size=1, use_bias=False, norm='bn'))

        if stride == 1 and is_last:
            self.xception_block.append(self.relu)
            self.xception_block.append(ConvLayer(out_dim, out_dim, kernel_size=3, norm='bn',
                                                 groups=out_dim, use_bias=False, fixed_padding=True))
            self.xception_block.append(ConvLayer(out_dim, out_dim, kernel_size=1, use_bias=False, norm='bn'))

        if not start_with_relu:
            self.xception_block = self.xception_block[1:]

        self.xception_block = nn.Sequential(*self.xception_block)

    def forward(self, x):
        residual = x
        out = self.xception_block(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return
