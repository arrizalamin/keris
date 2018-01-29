from keris.optimizers.optimizer import Optimizer
import numpy as np


class MomentumGD(Optimizer):
    def __init__(self, lr=1e-3, decay=0.9, mu=0.9, nesterov=False):
        self.lr = lr
        self.mu = mu
        self.decay = decay
        self.nesterov = nesterov
        self.param_configs = dict()

    def add_parameter(self, name, shape):
        self.param_configs[name]['v'] = np.zeros(shape)
        if self.nesterov:
            self.param_configs[name]['v_prev'] = np.zeros(shape)

    def update(self, name, x, dx):
        config = self.param_configs[name]
        if self.nesterov:
            config['v_prev'] = config['v']
            config['v'] = (self.mu * config['v']) - (self.lr * dx)

            out = x - (self.mu * config['v_prev']) + \
                (1 + self.mu) * config['v']
        else:
            config['v'] = (self.mu * config['v']) - (self.lr * dx)
            out = x - config['v']

        self.param_configs[name] = config

        return out
