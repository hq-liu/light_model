import torch
from torch import nn
import time


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


model1 = nn.Sequential(
    nn.Conv2d(in_channels=240, out_channels=60, kernel_size=1, groups=60),
    nn.Conv2d(in_channels=60, out_channels=60, kernel_size=1),
    nn.BatchNorm2d(60),
    nn.ReLU6(True),
    nn.Conv2d(in_channels=60, out_channels=60, kernel_size=3, groups=60, padding=1),
    nn.BatchNorm2d(60),
    nn.Conv2d(in_channels=60, out_channels=60, kernel_size=1),
    nn.Conv2d(in_channels=60, out_channels=240, kernel_size=1, groups=60)
)
conv1 = nn.Conv2d(in_channels=240, out_channels=60, kernel_size=1, groups=3)
model2 = nn.Sequential(
    nn.BatchNorm2d(60),
    nn.ReLU6(True),
    nn.Conv2d(in_channels=60, out_channels=60, kernel_size=3, groups=60, padding=1),
    nn.BatchNorm2d(60),
    nn.Conv2d(in_channels=60, out_channels=240, kernel_size=1, groups=3)
)

a = torch.randn(10, 240, 56, 56)
tic = time.time()
b = model1(a)
toc = time.time()
print(toc-tic)

tic = time.time()
c = conv1(a)
c = channel_shuffle(c, groups=3)
c = model2(c)
toc = time.time()
print(toc-tic)

