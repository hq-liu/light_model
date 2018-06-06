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


bottle_neck_channel=60
model1 = nn.Sequential(
                nn.Conv2d(in_channels=240, out_channels=bottle_neck_channel,
                          kernel_size=1, groups=bottle_neck_channel),
                nn.BatchNorm2d(bottle_neck_channel),
                nn.ReLU6(True),
                nn.Conv2d(in_channels=bottle_neck_channel,
                          out_channels=bottle_neck_channel,
                          kernel_size=1, groups=3),
                nn.BatchNorm2d(bottle_neck_channel),
                nn.ReLU6(True)
)
print(model1[3].weight.size())

model2 = nn.Sequential(
                nn.Conv2d(in_channels=240, out_channels=bottle_neck_channel,
                          kernel_size=1, groups=3),
                nn.BatchNorm2d(bottle_neck_channel),
                nn.ReLU6(True)
)
print(model2[0].weight.size())
a = torch.randn(10, 240, 56, 56)
tic = time.time()
b = model1(a)
toc = time.time()
print(toc-tic)

tic = time.time()
c = model2(a)
toc = time.time()
print(toc-tic)

