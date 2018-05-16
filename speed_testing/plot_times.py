import pandas as pd
from matplotlib import pyplot as plt
from shuffle_net.shuffle_net import ShuffleNet
import torch
from torch.autograd import Variable
import time
from speed_testing.layers_with_time import (
    Conv2dWithTime, BatchNorm2dWithTime,
    MaxPool2dWithTime, AvgPool2dWithTime,
    ReLUWithTime, LinearWithTime
)


def plot_times(times):
    """

    :type times: dict
    :return:
    """
    data_frame = pd.DataFrame.from_dict(times, orient='index')
    data_frame.rename(columns={0: 'test_time'}, inplace=True)
    plt.hist(data_frame['test_time'])
    plt.show()
    print(data_frame)


if __name__ == '__main__':
    a = torch.randn(10, 3, 224, 224)
    a = Variable(a)
    model = ShuffleNet()
    # checkpoint = torch.load('/home/lhq/PycharmProjects/light_model/shuffle_net/shufflenet.pth.tar',
    #                         map_location=lambda storage, loc: storage)
    # model.load_state_dict(checkpoint['state_dict'])
    tic = time.time()
    b = model(a)
    toc = time.time()
    times = {}
    types = [Conv2dWithTime, BatchNorm2dWithTime,
             MaxPool2dWithTime, AvgPool2dWithTime,
             ReLUWithTime, LinearWithTime]
    conv_count = 0
    for c in model.modules():
        if isinstance(c, LinearWithTime):
            conv_count += 1
        if type(c) in types:
            times.setdefault(c.name, 0)
            times[c.name] += c.times
    plot_times(times)
