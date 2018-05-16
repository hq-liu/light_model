"""
Add attribute times in each module
"""


from torch import nn
import time
import torch
from torch.autograd import Variable
from torch.nn import functional as F


class Conv2dWithTime(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dWithTime, self).__init__(in_channels, out_channels, kernel_size,
                                             stride, padding, dilation, groups, bias)
        self.name = 'conv2d'
        self.times = 0

    def forward(self, input):
        tic = time.time()
        output = F.conv2d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        toc = time.time()
        self.times = toc-tic
        return output


class BatchNorm2dWithTime(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(BatchNorm2dWithTime, self).__init__(num_features=num_features)
        self.times = 0

    def forward(self, input):
        tic = time.time()
        output = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                              self.training, self.momentum, self.eps)
        toc = time.time()
        self.times = toc - tic
        return output


class AvgPool2dWithTime(nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(AvgPool2dWithTime, self).__init__(kernel_size, stride, padding, ceil_mode, count_include_pad)
        self.times = 0

    def forward(self, input):
        tic = time.time()
        output = F.avg_pool2d(input, self.kernel_size, self.stride,
                              self.padding, self.ceil_mode, self.count_include_pad)
        toc = time.time()
        self.times = toc - tic
        return output


class MaxPool2dWithTime(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool2dWithTime, self).__init__(kernel_size=kernel_size, stride=stride, padding=padding,
                                                dilation=dilation, return_indices=return_indices,
                                                ceil_mode=ceil_mode)
        self.times = 0

    def forward(self, input):
        tic = time.time()
        output = F.max_pool2d(input, self.kernel_size, self.stride,
                              self.padding, self.dilation, self.ceil_mode,
                              self.return_indices)
        toc = time.time()
        self.times = toc - tic
        return output


class ReLUWithTime(nn.ReLU):
    def __init__(self, inplace=False):
        super(ReLUWithTime, self).__init__(inplace=inplace)
        self.times = 0

    def forward(self, input):
        tic = time.time()
        output = F.relu(input, self.inplace)
        toc = time.time()
        self.times = toc - tic
        return output


class LinearWithTime(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWithTime, self).__init__(in_features=in_features,
                                             out_features=out_features,
                                             bias=bias)
        self.times = 0

    def forward(self, input):
        tic = time.time()
        output = F.linear(input, self.weight, self.bias)
        toc = time.time()
        self.times = toc - tic
        return output


if __name__ == '__main__':
    a = torch.randn(1, 3, 224, 224)
    a = Variable(a)
    model = Conv2dWithTime(3, 4, kernel_size=3, padding=1)
    b = model(a)
    print(model.weight)
    print(model.times)

