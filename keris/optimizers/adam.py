from keris.optimizers.optimizer import Optimizer
import numpy as np


class Adam(Optimizer):
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 decay=0.9, stop_bias_correction_on_epoch=None):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.param_configs = dict()
        self.decay = decay
        self.epoch = 0
        self.stop_bias_correction = stop_bias_correction_on_epoch

    def add_parameter(self, name, shape):
        self.param_configs[name] = {
            'm': np.zeros(shape, dtype=np.float32),
            'v': np.zeros(shape, dtype=np.float32),
        }

    def update(self, name, x, dx):
        config = self.param_configs[name]
        self.epoch += 1

        config['m'] = (self.beta1 * config['m'] + (1 - self.beta1) * dx)
        config['v'] = (self.beta2 * config['v'] + (1 - self.beta2) * (dx ** 2))

        if type(self.stop_bias_correction) is int and self.epoch > self.stop_bias_correction:
            mb = config['m']
            vb = config['v']
        else:
            mb = config['m'] / (1 - self.beta1 ** self.epoch)
            vb = config['v'] / (1 - self.beta2 ** self.epoch)

        self.param_configs[name] = config
        out = x - self.lr * mb / (np.sqrt(vb) + self.epsilon)

        return out
