import torch
from torch import nn


class Model1(nn.Module):
    def __init__(self, num_classes=100):
        super(Model1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2),  # [24, 36, 36]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [24, 18, 18]
        )
        self.block = self._block()
        self.fc = nn.Linear(24*18*18, num_classes)

    @staticmethod
    def _block():
        return nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24 * 6, kernel_size=1),  # [24*6, 18, 18]
            nn.BatchNorm2d(24*6),
            nn.ReLU6(True),
            nn.Conv2d(in_channels=24*6, out_channels=24*6, kernel_size=3, padding=1, groups=24*6),
            nn.BatchNorm2d(24 * 6),
            nn.ReLU6(True),
            nn.Conv2d(in_channels=24*6, out_channels=24, kernel_size=1)  # [24, 18, 18]
        )

    def forward(self, x):
        x = self.conv(x)
        y = self.block(x)
        y += x
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y


if __name__ == '__main__':
    a = torch.randn(1, 3, 72, 72)
    model = Model1()
    b = model(a)
    print(b.size())
