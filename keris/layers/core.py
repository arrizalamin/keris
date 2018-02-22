from functools import reduce
from operator import mul
import keris.backend as K
from keris.layers.layer import Layer
import numpy as np


class Input(Layer):
    def __init__(self, input_shape, name):
        super().__init__(name)
        self.prev_layer = []
        self.shape = input_shape

    def forward(self, x, mode):
        return x

    def backward(self, dout, mode):
        return dout, None


class Dense(Layer):
    def __init__(self, units, name):
        super().__init__(name)
        self.trainable = True
        self.shape = units

    def _get_output_shape(self, input_shape):
        return self.shape

    def _initialize_params(self, input_shape, output_shape):
        if isinstance(input_shape, tuple):
            hidden = reduce(mul, list(input_shape), 1)
        else:
            hidden = input_shape
        output = self.shape
        params = {
            'w': self.weight_scale * self._with_initializer('random_normal', (hidden, output)),
            'b': K.zeros(output, K.float32),
        }
        return params

    def forward(self, x, mode):
        self.x = x
        w, b = self.params['w'], self.params['b']

        N = x.shape[0]
        x_rsp = x.reshape(N, -1)
        out = x_rsp.dot(w) + b

        return out

    def backward(self, dout, mode):
        x, w = self.x, self.params['w']

        N = x.shape[0]
        x_rsp = x.reshape(N, -1)
        dx = dout.dot(w.T)
        dx = dx.reshape(*x.shape)
        dw = x_rsp.T.dot(dout)
        db = dout.sum(axis=0)

        grads = {
            'w': dw,
            'b': db
        }
        return dx, grads
