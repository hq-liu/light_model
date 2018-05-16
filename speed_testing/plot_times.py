import pandas as pd
from matplotlib import pyplot as plt
from shuffle_net.shuffle_net_testing import ShuffleNet
import torch
from torch.autograd import Variable
import time
from speed_testing.layers_with_time import (
    Conv2dWithTime, BatchNorm2dWithTime,
    MaxPool2dWithTime, AvgPool2dWithTime,
    ReLUWithTime, LinearWithTime
)
from mobile_net.mobile_net import MobileNet
from mobile_net.mobile_net_v2 import MobileNetV2


def plot_times(model, model_name):
    """

    :type times: dict
    :return:
    """
    a = torch.randn(10, 3, 224, 224)
    a = Variable(a)
    tic = time.time()
    b = model(a)
    toc = time.time()
    times = {}
    types = [Conv2dWithTime, BatchNorm2dWithTime,
             MaxPool2dWithTime, AvgPool2dWithTime,
             ReLUWithTime, LinearWithTime]
    t_time = 0
    for c in model.modules():
        if type(c) in types:
            times.setdefault(c.name, 0)
            times[c.name] += c.times
            t_time += c.times
    print('-'*5+model_name+'-'*5)
    print('Total time: ', toc - tic)
    print('Total time2: ', t_time)

    data_frame = pd.DataFrame.from_dict(times, orient='index')
    data_frame.rename(columns={0: 'test_time'}, inplace=True)
    print(data_frame)
    print()


if __name__ == '__main__':
    shuffle_net = ShuffleNet()
    mobile_net = MobileNet()
    mobile_net_v2 = MobileNetV2()
    # checkpoint = torch.load('/home/lhq/PycharmProjects/light_model/shuffle_net/shufflenet.pth.tar',
    #                         map_location=lambda storage, loc: storage)
    # model.load_state_dict(checkpoint['state_dict'])
    plot_times(shuffle_net, 'shuffle_net')
    plot_times(mobile_net, 'mobile_net')
    plot_times(mobile_net, 'mobile_net_v2')
