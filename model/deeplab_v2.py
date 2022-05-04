import torch.nn as nn
import torch.nn.functional as F
import torch
from model.backbone import resnet, mobilenet, drn, vgg, xception
from model.backbone.layer_factory import ASPP_V2
from utils.net_util import weight_initialization, set_require_grad
from model.sync_batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

dilation_rates = [6, 12, 18, 24]
padding_series = [6, 12, 18, 24]
norm_instance = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                 SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d)


class DeebLab(nn.Module):

    def __init__(self, num_classes, initial='kaiming', backbone='resnet'):
        super(DeebLab, self).__init__()
        support_backbone = ['resnet', 'mobilenet', 'drn', 'vgg', 'xception']
        assert support_backbone.__contains__(backbone), 'unsupported backbone'

        if backbone == 'resnet':
            self.backbone = resnet.resnet_101(pre_trained=True)
        elif backbone == 'mobilenet':
            self.backbone = mobilenet.mobilenet_v2(pre_trained=True)
        elif backbone == 'drn':
            self.backbone = drn.drn_d_105(pre_trained=True)
        elif backbone == 'vgg':
            self.backbone = vgg.vgg_16(pre_trained=True)
        elif backbone == 'xception':
            self.backbone = xception.modified_xception(pre_trained=True)

        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                self.in_dim = m.weight.data.size()[0]

        self.dilation_rates = dilation_rates
        self.padding_series = padding_series
        self.classifier = weight_initialization(
            ASPP_V2(in_dim=self.in_dim, dilation_rates=dilation_rates, padding_rates=padding_series,
                    num_classes=num_classes), init_type=initial)
        self.classifier_aux = weight_initialization(
            ASPP_V2(in_dim=1024, dilation_rates=dilation_rates, padding_rates=padding_series,
                    num_classes=num_classes), init_type=initial)

        # freeze bn layers
        self.freeze_bn()

    def forward(self, x):
        h, w = x.size()[2:]
        feature_1, feature_2 = self.backbone(x)
        out1 = self.classifier_aux(feature_1)
        out2 = self.classifier(feature_2)
        out1 = F.interpolate(out1, size=(h, w), mode='bilinear', align_corners=True)
        out2 = F.interpolate(out2, size=(h, w), mode='bilinear', align_corners=True)
        return out1, out2

    def freeze_bn(self):
        """
        freeze the batch normalization layers
        """
        for module in self.modules():
            if isinstance(module, norm_instance):
                set_require_grad(module, False)

    def get_1x_lr_params(self):
        modules = []
        modules.append(self.backbone)

        for i in range(len(modules)):
            for j in modules[i].modules():
                if isinstance(j, nn.Conv2d) or isinstance(j, norm_instance):
                    for k in j.parameters():
                        if k.requires_grad:
                            yield k

    def get_10x_lr_params(self):
        modules = []
        modules.append(self.classifier)
        modules.append(self.classifier_aux)

        for i in range(len(modules)):
            for j in modules[i].modules():
                if isinstance(j, nn.Conv2d) or isinstance(j, norm_instance):
                    for k in j.parameters():
                        if k.requires_grad:
                            yield k

    def get_optim_params(self, parser):
        return [{'params': self.get_1x_lr_params(), 'lr': parser.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * parser.learning_rate}]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deeplab = DeebLab(num_classes=19, backbone='resnet')
    deeplab = deeplab.to(device)
    image = torch.ones(1, 3, 720, 1280)
    image = image.to(device)
    _, output = deeplab(image)
    print(output)
    print(output.size())
