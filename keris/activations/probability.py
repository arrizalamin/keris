import numpy as np
from layers.layer import Layer


class Softmax(Layer):

    """
    Compute softmax in a numerically stable way
    https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    """

    def __init__(self, name, axis=-1):
        super().__init__(name)
        self.axis = axis

    def forward(self, x, mode):
        out = np.exp(x - np.max(x, axis=self.axis, keepdims=True))
        out /= np.sum(out, axis=self.axis, keepdims=True)

        return out

    def backward(self, dout, mode):
        dx = np.exp(dout)
        return dx, None
