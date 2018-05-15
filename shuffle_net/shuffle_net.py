import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import time
import torchvision
from mobile_net.mobile_net import MobileNet
from torch.autograd import Variable


class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups=3,
                 grouped_conv=True):
        super(ShuffleNetUnit, self).__init__()
        bottleneck_channels = in_channels // 4
        self.groups = groups
        self.first_1x1_groups = self.groups if grouped_conv else 1
        self.stride = stride
        out_channels = out_channels-in_channels if stride == 2 else out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=bottleneck_channels,
                      kernel_size=1, stride=1, groups=self.first_1x1_groups),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=bottleneck_channels, out_channels=bottleneck_channels,
                      stride=stride, kernel_size=3, groups=bottleneck_channels, padding=1),
            nn.BatchNorm2d(bottleneck_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=bottleneck_channels, out_channels=out_channels,
                      stride=1, kernel_size=1, groups=groups),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x

        if self.stride == 2:
            residual = F.avg_pool2d(residual, kernel_size=3,
                                    stride=2, padding=1)
        x = self.conv1(x)
        x = self._channel_shuffle(x, self.groups)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.stride == 1:
            return F.relu(residual+x, inplace=True)
        elif self.stride == 2:
            return F.relu(torch.cat((residual, x), 1), inplace=True)

    @staticmethod
    def _channel_shuffle(feature_map, groups):
        N, C, H, W = feature_map.size()
        assert C % groups == 0, "Channel must be divisible by groups"
        channels_per_group = C // groups
        feature_map = feature_map.view(N, groups, channels_per_group, H, W)
        feature_map = torch.transpose(feature_map, 1, 2).contiguous()
        feature_map = feature_map.view(N, -1, H, W)
        return feature_map


class ShuffleNet(nn.Module):
    def __init__(self, groups=3, in_channels=3, num_classes=1000):
        super(ShuffleNet, self).__init__()
        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels = in_channels
        self.num_classes = num_classes

        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(groups))
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.stage_out_channels[1],
                               kernel_size=3, stride=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage_2 = self._make_stage(2)
        self.stage_3 = self._make_stage(3)
        self.stage_4 = self._make_stage(4)
        self.global_pool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(in_features=self.stage_out_channels[-1], out_features=num_classes)

    def _make_stage(self, stage):
        modules = OrderedDict()
        stage_name = "ShuffleUnit_Stage{}".format(stage)

        # First ShuffleUnit in the stage
        # 1. non-grouped 1x1 convolution (i.e. pointwise convolution)
        #   is used in Stage 2. Group convolutions used everywhere else.
        grouped_conv = stage > 2

        # 2. concatenation unit is always used.
        first_module = ShuffleNetUnit(
            in_channels=self.stage_out_channels[stage - 1],
            out_channels=self.stage_out_channels[stage],
            groups=self.groups,
            grouped_conv=grouped_conv,
            stride=2
        )
        modules[stage_name + "_0"] = first_module

        # add more ShuffleUnits depending on pre-defined number of repeats
        for i in range(self.stage_repeats[stage - 2]):
            name = stage_name + "_{}".format(i + 1)
            module = ShuffleNetUnit(
                self.stage_out_channels[stage],
                self.stage_out_channels[stage],
                groups=self.groups,
                grouped_conv=True,
                stride=1
            )
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def cul_time(model, batch_size):
    a = torch.randn(batch_size, 3, 224, 224)
    a = Variable(a)
    tic = time.time()
    b = model(a)
    toc = time.time()
    print(int(round((toc-tic)*1000)))
    print(b.size())


if __name__ == '__main__':
    # unit = ShuffleNetUnit(in_channels=240, out_channels=480, stride=2)
    alex = torchvision.models.alexnet()
    model = ShuffleNet()
    mobile_net = MobileNet(1000)
    cul_time(model, 10)
    cul_time(mobile_net, 10)
    cul_time(alex, 10)


