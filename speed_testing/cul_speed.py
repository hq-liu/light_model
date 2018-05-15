from torch import nn
import time


class CulTime(object):
    def __init__(self):
        self.times = {}

    def reset(self):
        self.times = {}

    def update(self, update_part, update_time):
        self.times.setdefault(update_part, 0)
        self.times[update_part] += update_time


if __name__ == '__main__':
    ct = CulTime()
    ct.update('conv_dw', 0.01)
    print(ct.times['conv_dw'])
