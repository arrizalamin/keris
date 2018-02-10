import numpy as np
from keris.layers.layer import Layer
from keris.utils.im2col import im2col, col2im


class Conv2D(Layer):
    def __init__(self, filters, kernel_size, name, stride=1,
                 padding='valid'):
        super().__init__(name)
        self.trainable = True
        self.kernel_size, self.filters = kernel_size, filters
        self.stride, self.padding = stride, padding
        self.pad = int(filters[0] - 1) // 2 if padding == 'same' else 0

    def _get_output_shape(self, input_shape):
        _, H, W = input_shape
        pad, stride = self.pad, self.stride
        (filter_H, filter_W) = self.filters
        height = (H + 2 * pad - filter_H) // stride + 1
        width = (W + 2 * pad - filter_W) // stride + 1

        return (self.kernel_size, height, width)

    def _initialize_params(self, input_shape, output_shape):
        channels = input_shape[0]
        params = {
            'w': self._with_initializer('glorot_normal', (self.kernel_size,
                                                          channels,
                                                          *self.filters)),

            'b': np.zeros(self.kernel_size, dtype=np.float32)
        }
        return params

    def forward(self, x, mode):
        self.x, stride, pad = x, self.stride, self.pad
        w, b = self.params['w'], self.params['b']
        N, _, H, W = x.shape
        num_filters, _, filter_height, filter_width = w.shape

        # Check dimensions
        # assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
        # assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

        # Create output
        out_height = (H + 2 * pad - filter_height) // stride + 1
        out_width = (W + 2 * pad - filter_width) // stride + 1

        if x.dtype == np.float64:
            print(self.name)

        self.x_cols = x_cols = np.asarray(
            im2col(x, filter_height, filter_width, pad, stride))
        res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

        out = res.reshape(self.kernel_size, out_height, out_width, N)
        out = out.transpose(3, 0, 1, 2)

        return out

    def backward(self, dout, mode):
        x, w = self.x, self.params['w']
        stride, pad, x_cols = self.stride, self.pad, self.x_cols
        N, C, H, W = x.shape

        db = np.sum(dout, axis=(0, 2, 3))

        num_filters, _, filter_height, filter_width = w.shape
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
        dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

        dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
        dx = np.asarray(
            col2im(dx_cols, N, C, H, W, filter_height, filter_width, pad,
                   stride))

        grads = {
            'w': dw,
            'b': db,
        }

        return dx, grads


class Conv1x1NoPad(Layer):
    def __init__(self, kernel_size, name):
        super().__init__(name)
        self.trainable = True
        self.kernel_size = kernel_size

    def _initialize_params(self, input_shape, output_shape):
        C = input_shape[0]
        out = self.kernel_size
        params = {
            'w': self._with_initializer('glorot_normal', (out, C, 1, 1)),
            'b': self.weight_scale * np.zeros(out),
        }

        return params

    def _get_output_shape(self, input_shape):
        C, H, W = input_shape
        out = self.kernel_size
        out_shape = out, H, W

        return out_shape

    def forward(self, x, mode):
        self.x, w, b = x, self.params['w'], self.params['b']
        N, _, H, W = x.shape
        F = w.shape[0]
        out = np.zeros((N, F, H, W))
        for i in range(F):
            out[:, i, :, :] = np.sum(x * w[i], axis=1) + b[i]

        return out

    def backward(self, dout, mode):
        x, w = self.x, self.params['w']
        N, _, H, W = x.shape
        F = w.shape[0]

        db = dout.sum(axis=(0, 2, 3))
        dw = np.zeros_like(w)
        dx = np.zeros_like(x)

        for i in range(F):
            dw[i, :, :, :] = np.sum(x * dout[:, i, :, :][:, None],
                                    axis=(0, 2, 3))[:, None, None]
        for i in range(N):
            dx[i, :, :, :] = (w * dout[i, :, :, :][:, None]).sum(axis=0)

        grads = {
            'w': dw,
            'b': db,
        }
        return dx, grads
