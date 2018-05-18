import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import time
import torchvision
from speed_testing.layers_with_time import (
    Conv2dWithTime, BatchNorm2dWithTime,
    MaxPool2dWithTime, AvgPool2dWithTime,
    ReLU6WithTime, LinearWithTime
)


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()
        self.base_model = nn.Sequential(
            self._conv_bn(3, 32, 2, 'entry'),
            self._conv_dw(32, 64, 1, '32_64'),
            self._conv_dw(64, 128, 2, '64_128'),
            self._conv_dw(128, 128, 1, '128_128'),
            self._conv_dw(128, 256, 2, '128_256'),
            self._conv_dw(256, 256, 1, '256_256'),
            self._conv_dw(256, 512, 2, '256_256'),
            self._conv_dw(512, 512, 1, '512_512'),
            self._conv_dw(512, 512, 1, '512_512'),
            self._conv_dw(512, 512, 1, '512_512'),
            self._conv_dw(512, 512, 1, '512_512'),
            self._conv_dw(512, 512, 1, '512_512'),
            self._conv_dw(512, 1024, 2, '512_1024'),
            self._conv_dw(1024, 1024, 1, '1024_1024'),
        )
        self.global_pool = AvgPool2dWithTime(kernel_size=7)
        self.global_pool.name = 'global_pool'+'_'+self.global_pool.layer_type
        self.fc = LinearWithTime(1024, num_classes)
        self.fc.name = 'fc'+'_'+self.fc.layer_type

    @staticmethod
    def _conv_bn(in_channels, out_channels, stride, layer_name):
        conv = Conv2dWithTime(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, stride=stride, padding=1, bias=False)
        bn = BatchNorm2dWithTime(out_channels)
        relu = ReLU6WithTime(inplace=True)
        conv.name, bn.name, relu.name = layer_name+'_'+conv.layer_type, \
                                        layer_name+'_'+bn.layer_type, \
                                        layer_name+'_'+relu.layer_type
        return nn.Sequential(
            conv, bn, relu
        )

    @staticmethod
    def _conv_dw(in_channels, out_channels, stride, layer_name):
        conv_dw = Conv2dWithTime(in_channels=in_channels, out_channels=in_channels,
                                 kernel_size=3, stride=stride, padding=1, groups=in_channels,
                                 bias=False)
        bn = BatchNorm2dWithTime(in_channels)
        relu_1 = ReLU6WithTime(inplace=True)
        conv_pw = Conv2dWithTime(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=1, stride=1, padding=0)
        relu_2 = ReLU6WithTime(inplace=True)
        conv_dw.name, bn.name, relu_1.name, conv_pw.name, relu_2.name = layer_name+'_'+conv_dw.layer_type, \
                                                                        layer_name+'_'+bn.layer_type, \
                                                                        layer_name+'_'+relu_1.layer_type+'_after_dw', \
                                                                        layer_name+'_'+conv_pw.layer_type, \
                                                                        layer_name+'_'+relu_2.layer_type+'_after_pw'
        return nn.Sequential(
            conv_dw, bn, relu_1, conv_pw, relu_2
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.global_pool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    a = torch.randn(10, 3, 224, 224)
    a = Variable(a)
    model = MobileNet(1000)
    b = model(a)
    print(b.size())
