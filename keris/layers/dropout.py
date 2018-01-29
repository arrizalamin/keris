import numpy as np
from keris.layers.layer import Layer


class Dropout(Layer):
    def __init__(self, rate, name):
        super().__init__(name)
        self.rate = rate

    def forward(self, x, mode):
        out = None
        if mode == 'train':
            self.mask = np.random.rand(*x.shape) >= self.rate
            out = x * self.mask
        elif mode == 'test':
            out = x * self.rate

        out = out.astype(x.dtype, copy=False)
        return out

    def backward(self, dout, mode):
        dx = None
        if mode == 'train':
            dx = dout * self.mask
        elif mode == 'test':
            dx = dout
        return dx, None
