import numpy as np


class Optimizer:
    def add_parameter(self, name, shape):
        pass

    def update(self, name, x, dx):
        raise NotImplementedError

    def decrease_lr(self):
        self.lr *= self.decay
