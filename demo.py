from shuffle_net.shuffle_net import ShuffleNet
import torch
from torch.autograd import Variable


model = ShuffleNet()

a = torch.randn(1, 3, 224, 224)
a = Variable(a)
b = model(a)
print(b)
