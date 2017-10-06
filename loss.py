from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        return np.mean(np.sum((input - target) ** 2, 1) / 2.)

    def backward(self, input, target):
        return (input - target) / target.shape[0]
