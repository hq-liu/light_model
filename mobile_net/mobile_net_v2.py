import torch.nn as nn
import math
from speed_testing.layers_with_time import (
    Conv2dWithTime, BatchNorm2dWithTime,
    MaxPool2dWithTime, AvgPool2dWithTime,
    ReLUWithTime, LinearWithTime, ReLU6WithTime,
    DropoutWithTime
)
import torch
from torch.autograd import Variable


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        Conv2dWithTime(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2dWithTime(oup),
        ReLU6WithTime(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        Conv2dWithTime(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2dWithTime(oup),
        ReLU6WithTime(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            Conv2dWithTime(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            BatchNorm2dWithTime(inp * expand_ratio),
            ReLU6WithTime(inplace=True),
            # dw
            Conv2dWithTime(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            BatchNorm2dWithTime(inp * expand_ratio),
            ReLU6WithTime(inplace=True),
            # pw-linear
            Conv2dWithTime(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            BatchNorm2dWithTime(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(AvgPool2dWithTime(input_size//32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            DropoutWithTime(),
            LinearWithTime(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    a = torch.randn(1, 3, 224, 224)
    a = Variable(a)
    model = MobileNetV2()
    model.load_state_dict(torch.load('mobilenetv2_718.pth.tar', map_location=lambda storage, loc: storage))
    b = model(a)
    cnt = 0
    for c in model.modules():
        if isinstance(c, ReLUWithTime):
            print(c.name)

    print(b, cnt)

