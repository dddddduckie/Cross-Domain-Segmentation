import torch.nn as nn
import torch.nn.functional as F
from model.backbone.layer_factory import ConvLayer


class Discriminator(nn.Module):

    def __init__(self, num_classes, dim=64):
        super(Discriminator, self).__init__()

        self.conv1 = ConvLayer(in_dim=num_classes, out_dim=dim, kernel_size=5, stride=2,
                               padding=1, activation='lrelu')
        self.conv2 = ConvLayer(in_dim=dim, out_dim=dim * 2, kernel_size=3, stride=1,
                               padding=1, activation='lrelu')
        self.conv3 = ConvLayer(in_dim=dim * 2, out_dim=dim * 4, kernel_size=3, stride=1,
                               padding=1, activation='lrelu')
        self.conv4 = ConvLayer(in_dim=dim * 4, out_dim=dim * 8, kernel_size=3, stride=1,
                               padding=1, activation='lrelu')
        self.classifier = nn.Conv2d(dim * 8, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # h, w = x.size()[2:]
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.classifier(out)
        # out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        return out
