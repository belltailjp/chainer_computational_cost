import unittest

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

import chainer_computational_cost


class SimpleConvNet(chainer.Chain):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1)
            self.bn1 = L.BatchNormalization(32)
            self.conv2 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1)
            self.bn2 = L.BatchNormalization(32)
            self.conv3 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1)
            self.bn3 = L.BatchNormalization(32)
            self.fc4 = L.Linear(None, 100)
            self.fc5 = L.Linear(None, 10)

    def __call__(self, h):
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        height, width = h.shape[2:]
        h = F.average_pooling_2d(h, ksize=(height, width))
        h = F.reshape(h, (h.shape[0], -1))
        h = F.relu(self.fc4(h))
        return self.fc5(h)


def test_simple_net():
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    net = SimpleConvNet()
    with chainer.using_config('train', False):
        with chainer_computational_cost.ComputationalCostHook() as cost:
            net(x)

