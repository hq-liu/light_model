from torch import nn
from speed_testing.cul_speed import CulTime
import time
import torch
from torch.autograd import Variable


class ConvWithTime(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, layer_name, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(ConvWithTime, self).__init__()
        self.times = 0
        self.layer_name = layer_name
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias, groups=groups, dilation=dilation
                              )

    def forward(self, x):
        tic = time.time()
        x = self.conv(x)
        toc = time.time()
        self.times = toc-tic
        return x


class BNWithTime(nn.Module):
    def __init__(self, num_features, layer_name):
        super(BNWithTime, self).__init__()
        self.times = 0
        self.layer_name = layer_name
        self.conv = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x):
        tic = time.time()
        x = self.conv(x)
        toc = time.time()
        self.times = toc-tic
        return x


class AvgPoolWithTime(nn.Module):
    def __init__(self, kernel_size, layer_name, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(AvgPoolWithTime, self).__init__()
        self.times = 0
        self.layer_name = layer_name
        self.conv = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding,
                                 ceil_mode=ceil_mode,
                                 count_include_pad=count_include_pad)

    def forward(self, x):
        tic = time.time()
        x = self.conv(x)
        toc = time.time()
        self.times = toc-tic
        return x


class MaxPoolWithTime(nn.Module):
    def __init__(self, kernel_size, layer_name, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPoolWithTime, self).__init__()
        self.times = 0
        self.layer_name = layer_name
        self.conv = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation,
                                 return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, x):
        tic = time.time()
        x = self.conv(x)
        toc = time.time()
        self.times = toc-tic
        return x


class MultiConv(nn.Module):
    def __init__(self):
        super(MultiConv, self).__init__()
        self.times = {}
        layer_classes = ['bn', 'conv']
        self.conv = nn.Sequential(
            ConvWithTime(3, 40, 3, 'conv'),
            BNWithTime(40, 'bn'),
            ConvWithTime(40, 40, 3, 'conv'),
            BNWithTime(40, 'bn'),
            ConvWithTime(40, 40, 3, 'conv'),
            BNWithTime(40, 'bn'),
            ConvWithTime(40, 40, 3, 'conv')
        )
        for i in layer_classes:
            self.times[i] = 0

    def forward(self, x):
        x = self.conv(x)
        for i in model.children():
            if isinstance(i, nn.Sequential):
                for j in i:
                    self.times[j.layer_name] += j.times
        return x


if __name__ == '__main__':
    a = torch.randn(1, 3, 224, 224)
    a = Variable(a)
    model = MultiConv()
    b = model(a)
    print(model.times['conv'], model.times['bn'])

