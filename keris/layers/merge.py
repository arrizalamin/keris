import numpy as np
from keris.layers.layer import Layer


class Concatenate(Layer):
    def __init__(self, axis, name):
        super().__init__(name)
        self.axis = axis

    def _check_input(self, x):
        if not isinstance(x, (list, tuple)):
            raise Exception('Input to this layer must be a list/tuple')

    def _get_output_shape(self, input_shape):
        self._check_input(input_shape)
        axis = self.axis
        self.in_axis = []

        last_axis = 0
        for in_shape in input_shape:
            last_axis += in_shape[axis]
            self.in_axis.append(last_axis)

        out = list(input_shape[0])
        for shape in input_shape[1:]:
            out[axis] += shape[axis]

        return tuple(out)

    def forward(self, x, mode):
        self._check_input(x)
        out = np.concatenate(tuple(x), axis=self.axis + 1)
        return out

    def backward(self, dout, mode):
        dx = np.split(dout, self.in_axis, axis=self.axis + 1)
        return tuple(dx), None


class Sum(Layer):
    def _check_input(self, x):
        if not isinstance(x, (list, tuple)):
            raise Exception('Input to this layer must be a list/tuple')

    def _get_output_shape(self, input_shape):
        self._check_input(input_shape)
        self.total_input = len(input_shape)
        return input_shape[0]

    def forward(self, x, mode):
        self._check_input(x)
        out = np.sum(x, axis=0)
        return out

    def backward(self, dout, mode):
        dx = [dout for _ in range(self.total_input)]
        return tuple(dx), None
