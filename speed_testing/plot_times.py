import pandas as pd
from matplotlib import pyplot as plt
from shuffle_net.shuffle_net_testing import ShuffleNet
import torch
from torch.autograd import Variable
import time
from speed_testing.layers_with_time import (
    Conv2dWithTime, BatchNorm2dWithTime,
    MaxPool2dWithTime, AvgPool2dWithTime,
    ReLUWithTime, LinearWithTime, DropoutWithTime,
    ReLU6WithTime
)
from mobile_net.mobile_net_testing import MobileNet
from mobile_net.mobile_net_v2 import MobileNetV2
from mobile_net.mobile_net_v2_alt import MobileNetV2_alt
from mobile_net.mobile_net_v2_se import MobileNetV2_SE
from torch import nn


pd.set_option('display.max_rows', None)


def plot_times(model, model_name):
    a = torch.randn(50, 3, 224, 224)
    a = Variable(a)
    tic = time.time()
    b = model(a)
    toc = time.time()
    times = {}
    count = {}
    types = [Conv2dWithTime, BatchNorm2dWithTime,
             MaxPool2dWithTime, AvgPool2dWithTime,
             ReLUWithTime, LinearWithTime, ReLU6WithTime,
             DropoutWithTime]
    t_time = 0
    for c in model.modules():
        if type(c) in types:
            times.setdefault(c.name, 0)
            count.setdefault(c.name, 0)
            times[c.name] += c.run_time
            count[c.name] += 1
            t_time += c.run_time

    print('-'*5+model_name+'-'*5)
    # print('Total time: ', toc - tic)
    print('Total time cost: ', t_time)

    data_frame = pd.DataFrame.from_dict(times, orient='index')
    data_frame.rename(columns={0: 'test_time'}, inplace=True)
    print(data_frame)
    print()
    print('sorted_time cost')
    data_frame = data_frame.sort_index(by='test_time', ascending=False)
    print(data_frame)
    print()

    for c in count.keys():
        times[c] = times[c]/count[c]
    data_frame = pd.DataFrame.from_dict(times, orient='index')
    data_frame.rename(columns={0: 'test_time'}, inplace=True)
    print('Average time cost:')
    print(data_frame)
    print()
    print('sorted_time avg_cost:')
    data_frame = data_frame.sort_index(by='test_time', ascending=False)
    print(data_frame)
    print()


def cul_module_run_time(module_name, model, tensor):
    tic = time.time()
    tensor = model(tensor)
    toc = time.time()
    print(module_name+': Run time: {:.6f}'.format(toc-tic))


if __name__ == '__main__':
    shuffle_net = ShuffleNet()
    mobile_net = MobileNet()
    mobile_net_v2 = MobileNetV2()
    # plot_times(mobile_net, 'mobile_net')
    # plot_times(mobile_net, 'mobile_net')
    plot_times(shuffle_net, 'shuffle_net')
    plot_times(mobile_net, 'mobile_net')
    plot_times(mobile_net_v2, 'mobile_net_v2')

    # a = torch.randn(10, 50, 200, 200)
    # a = Variable(a)
    # cul_module_run_time('dw_s1', nn.Conv2d(50, 50, kernel_size=3, groups=50, stride=1), a)
    # cul_module_run_time('dw_s2', nn.Conv2d(50, 50, kernel_size=3, groups=50, stride=2, padding=1), a)
    # cul_module_run_time('pw', nn.Conv2d(50, 100, kernel_size=1, groups=1, stride=1), a)
    # cul_module_run_time('conv_s1', nn.Conv2d(50, 100, kernel_size=5, groups=1, stride=1), a)
    # cul_module_run_time('conv_s2', nn.Conv2d(50, 100, kernel_size=3, groups=1, stride=2, padding=1), a)
    # cul_module_run_time('relu', nn.ReLU(), a)
    # cul_module_run_time('relu(inplace)', nn.ReLU(True), a)
    # cul_module_run_time('relu6', nn.ReLU6(), a)
    # cul_module_run_time('relu6(inplace)', nn.ReLU6(True), a)
    # cul_module_run_time('elu', nn.ELU(), a)
    # cul_module_run_time('elu(inplace)', nn.ELU(inplace=True), a)
    # cul_module_run_time('conv_dw', nn.Conv2d(50, 50 ,3, groups=50), a)
    # cul_module_run_time('conv_1', nn.Conv2d(50, 100, 1), a)

