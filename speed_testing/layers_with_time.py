"""
Add attribute run_time\layer_type\name in each module
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
        if groups == in_channels:
            self.layer_type = 'conv2d_' + str(kernel_size) + '_dw'
        else:
            self.layer_type = 'conv2d_' + str(kernel_size) + '_' + str(groups)
        self.name = self.layer_type
        self.run_time = 0

    def forward(self, input):
        tic = time.time()
        output = F.conv2d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        toc = time.time()
        self.run_time = toc-tic
        return output


class BatchNorm2dWithTime(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(BatchNorm2dWithTime, self).__init__(num_features=num_features)
        self.layer_type = 'bn'
        self.name = self.layer_type
        self.run_time = 0

    def forward(self, input):
        tic = time.time()
        output = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                              self.training, self.momentum, self.eps)
        toc = time.time()
        self.run_time = toc - tic
        return output


class AvgPool2dWithTime(nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(AvgPool2dWithTime, self).__init__(kernel_size, stride, padding,
                                                ceil_mode, count_include_pad)
        self.layer_type = 'avg_pool'
        self.name = self.layer_type
        self.run_time = 0

    def forward(self, input):
        tic = time.time()
        output = F.avg_pool2d(input, self.kernel_size, self.stride,
                              self.padding, self.ceil_mode, self.count_include_pad)
        toc = time.time()
        self.run_time = toc - tic
        return output


class MaxPool2dWithTime(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool2dWithTime, self).__init__(kernel_size=kernel_size, stride=stride, padding=padding,
                                                dilation=dilation, return_indices=return_indices,
                                                ceil_mode=ceil_mode)
        self.layer_type = 'max_pool'
        self.name = self.layer_type
        self.run_time = 0

    def forward(self, input):
        tic = time.time()
        output = F.max_pool2d(input, self.kernel_size, self.stride,
                              self.padding, self.dilation, self.ceil_mode,
                              self.return_indices)
        toc = time.time()
        self.run_time = toc - tic
        return output


class ReLUWithTime(nn.ReLU):
    def __init__(self, inplace=False):
        super(ReLUWithTime, self).__init__(inplace=inplace)
        self.layer_type = 'relu'
        self.name = self.layer_type
        self.run_time = 0

    def forward(self, input):
        tic = time.time()
        output = F.relu(input, self.inplace)
        toc = time.time()
        self.run_time = toc - tic
        return output


class ReLU6WithTime(nn.ReLU6):
    def __init__(self, inplace=False):
        super(ReLU6WithTime, self).__init__(inplace=inplace)
        self.layer_type = 'relu6'
        self.name = self.layer_type
        self.run_time = 0

    def forward(self, input):
        tic = time.time()
        output = F.relu6(input, self.inplace)
        toc = time.time()
        self.run_time = toc - tic
        return output


class DropoutWithTime(nn.Dropout):
    def __init__(self, p=0.5, inplace=False):
        super(DropoutWithTime, self).__init__(p=p, inplace=inplace)
        self.layer_type = 'dropout'
        self.name = self.layer_type
        self.run_time = 0

    def forward(self, input):
        tic = time.time()
        output = F.dropout(input, self.p, self.training, self.inplace)
        toc = time.time()
        self.run_time = toc - tic
        return output


class LinearWithTime(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWithTime, self).__init__(in_features=in_features,
                                             out_features=out_features,
                                             bias=bias)
        self.layer_type = 'linear'
        self.name = self.layer_type
        self.run_time = 0

    def forward(self, input):
        tic = time.time()
        output = F.linear(input, self.weight, self.bias)
        toc = time.time()
        self.run_time = toc - tic
        return output


if __name__ == '__main__':
    a = torch.randn(1, 100)
    a = Variable(a)
    model = DropoutWithTime()
    b = model(a)
    types = [Conv2dWithTime, BatchNorm2dWithTime,
             MaxPool2dWithTime, AvgPool2dWithTime,
             ReLUWithTime, LinearWithTime]
    print(type(model) in types)

