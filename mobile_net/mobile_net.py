import torch
from torch import nn
from torch.nn import functional as F
import time
import torchvision


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()
        self.base_model = nn.Sequential(
            self._conv_bn(3, 32, 2),
            self._conv_dw(32, 64, 1),
            self._conv_dw(64, 128, 2),
            self._conv_dw(128, 128, 1),
            self._conv_dw(128, 256, 2),
            self._conv_dw(256, 256, 1),
            self._conv_dw(256, 512, 2),
            self._conv_dw(512, 512, 1),
            self._conv_dw(512, 512, 1),
            self._conv_dw(512, 512, 1),
            self._conv_dw(512, 512, 1),
            self._conv_dw(512, 512, 1),
            self._conv_dw(512, 1024, 2),
            self._conv_dw(1024, 1024, 1),
        )
        self.global_pool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(1024, num_classes)

    @staticmethod
    def _conv_bn(in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def _conv_dw(in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=3, stride=stride, padding=1, groups=in_channels,
                      bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.global_pool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    a = torch.randn(10, 3, 224, 224)
    model = MobileNet(1000)
    b = model(a)
    print(b.size())
