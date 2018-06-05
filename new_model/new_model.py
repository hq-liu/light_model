import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class NewModelBlock(nn.Module):
    def __init__(self, inp, oup, stride):
        super(NewModelBlock, self).__init__()
        self.stride = stride
        assert self.stride in [1, 2]
        self.bottle_neck_channel = oup // 4
        self.oup = oup if stride == 1 else oup-inp
        if stride == 1:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=inp, out_channels=self.bottle_neck_channel,
                          kernel_size=1, groups=self.bottle_neck_channel),
                nn.Conv2d(in_channels=self.bottle_neck_channel,
                          out_channels=self.bottle_neck_channel,
                          kernel_size=1, groups=3),
                nn.BatchNorm2d(self.bottle_neck_channel),
                nn.ReLU6(True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=inp, out_channels=self.bottle_neck_channel,
                          kernel_size=1, groups=3),
                nn.BatchNorm2d(self.bottle_neck_channel),
                nn.ReLU6(True)
            )
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.bottle_neck_channel,
                      out_channels=self.bottle_neck_channel,
                      kernel_size=3, groups=self.bottle_neck_channel,
                      stride=stride, padding=1),
            nn.BatchNorm2d(self.bottle_neck_channel)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.bottle_neck_channel,
                      out_channels=self.oup, kernel_size=1,
                      groups=3),
            nn.BatchNorm2d(self.oup)
        )

    def _combine_function(self, x, y):
        if self.stride == 1:
            return x+y
        else:
            x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
            return torch.cat((x, y), dim=1)

    @staticmethod
    def channel_shuffle(x, groups):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups,
                   channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x

    def forward(self, x):
        y = self.conv1(x)
        y = self.channel_shuffle(y, 3)
        y = self.dw_conv(y)
        y = self.conv2(y)
        return F.relu6(self._combine_function(x, y))


class NewModel(nn.Module):
    def __init__(self, num_classes=1000, in_channels=3):
        super(NewModel, self).__init__()
        self.stage_repeats = [3, 7, 3]
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stage_out_channels = [-1, 24, 240, 480, 960]
        self.conv1 = nn.Conv2d(self.in_channels, self.stage_out_channels[1],
                               kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Stage 2
        self.stage2 = self._make_stage(2)
        # Stage 3
        self.stage3 = self._make_stage(3)
        # Stage 4
        self.stage4 = self._make_stage(4)
        num_inputs = self.stage_out_channels[-1]
        self.fc = nn.Linear(num_inputs, self.num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_stage(self, stage):
        modules = list()
        stage_name = "ShuffleUnit_Stage{}".format(stage)

        # 2. concatenation unit is always used.
        first_module = NewModelBlock(self.stage_out_channels[stage - 1],
                                     self.stage_out_channels[stage],
                                     stride=2)
        modules.append(first_module)

        # add more ShuffleUnits depending on pre-defined number of repeats
        for i in range(self.stage_repeats[stage - 2]):
            module = NewModelBlock(self.stage_out_channels[stage],
                                   self.stage_out_channels[stage],
                                   stride=1)
            modules.append(module)

        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # global average pooling layer
        x = F.avg_pool2d(x, x.data.size()[-2:])

        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    a = torch.randn(1, 3, 224, 224)
    model = NewModel()
    b = model(a)
    print(b.size())
