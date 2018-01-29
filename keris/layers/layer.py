import numpy as np
from keris.initializers import initializers


class Layer:
    def __init__(self, name):
        self.trainable = False
        self.weight_scale = 1e-1
        self.prev_layers = []
        self.next_layers = []
        self.name = name

    def __call__(self, input_layer):
        if not isinstance(input_layer, (tuple, list)):
            # Link this layer to previous layer and vice versa
            input_layer._link_next_layer(self)
            self._link_prev_layer(input_layer)

            in_shape = input_layer.shape
        else:
            in_shape = []
            for layer in input_layer:
                layer._link_next_layer(self)
                self._link_prev_layer(layer)

                in_shape.append(layer.shape)

        # Calculate output shape of this layer
        out_shape = self._get_output_shape(input_shape=in_shape)
        self.shape = out_shape

        # Initialize parameters if layer is trainable
        if self.trainable:
            self.params = self._initialize_params(in_shape, out_shape)

        return self

    def _link_next_layer(self, layer):
        self.next_layers.append(layer)

    def _link_prev_layer(self, layer):
        self.prev_layers.append(layer)

    def _get_output_shape(self, input_shape):
        return input_shape

    def _initialize_params(self, input_shape, output_shape):
        raise NotImplementedError

    def _with_initializer(self, name, shape, dtype=np.float32):
        if name not in initializers:
            raise ValueError("initializer does not exists")

        return initializers[name](shape, dtype)

    def update_param(self, key, val):
        self.params[key] = val

    def forward(self, x, mode):
        raise NotImplementedError

    def backward(self, dout, mode):
        raise NotImplementedError
