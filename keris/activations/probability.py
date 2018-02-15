import numpy as np
from keris.layers.layer import Layer


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
        self.x, self.out = x, out

        return out

    def backward(self, dout, mode):
        """
        https://stackoverflow.com/a/40576872
        """
        x, out = self.x, self.out
        dx = np.zeros_like(dout)

        for i in range(len(out)):
            for j in range(len(x)):
                if i == j:
                    dx[i, j] = out[i] * (1 - x[i])
                else:
                    dx[i, j] = -out[i] * x[j]
        return dx, None
