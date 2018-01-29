from keris.optimizers.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, lr=1e-3, decay=0.9):
        self.lr = lr
        self.param_configs = dict()
        self.epsilon = 1e-8
        self.decay = decay

    def update(self, name, x, dx):
        out = x - (self.lr * dx)

        return out

    def decrease_lr(self):
        self.lr *= self.decay
