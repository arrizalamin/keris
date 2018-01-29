import numpy as np
from keris.layers.layer import Layer


class ReLU(Layer):
    def forward(self, x, mode):
        self.mask = x > 0
        out = x * self.mask

        return out

    def backward(self, dout, mode):
        dx = dout * self.mask
        return dx, None


class LeakyReLU(Layer):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, x, mode):
        out = x.copy()
        self.negative = negative = x < 0
        out[negative] *= 1e-2
        return out

    def backward(self, dout, mode):
        negative, dx = self.negative, dout.copy(),
        dx[negative] *= 1e-2
        return dx, None


class ELU(Layer):
    def forward(self, x, mode):
        self.x, out = x, x.copy()
        negative = x < 0
        out[negative] = 1 * (np.exp(x[negative]) - 1)
        return out

    def backward(self, dout, mode):
        x, dx = self.x, dout.copy()
        negative = x < 0
        dx[negative] *= 1 * np.exp(x[negative])
        return dx, None
