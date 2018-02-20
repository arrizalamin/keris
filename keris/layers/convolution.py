import keris.backend as K
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

            'b': K.zeros(self.kernel_size, dtype=K.float32)
        }
        return params

    def forward(self, x, mode):
        self.x_shape = x.shape
        N, C, H, W = x.shape
        stride, pad = self.stride, self.pad
        w, b = self.params['w'], self.params['b']
        _, _, filter_height, filter_width = w.shape

        # Check dimensions
        # assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
        # assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

        # Create output
        out_height = (H + 2 * pad - filter_height) // stride + 1
        out_width = (W + 2 * pad - filter_width) // stride + 1

        self.x_cols = x_cols = self._im2col(x)
        res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

        out = res.reshape(self.kernel_size, out_height, out_width, N)
        out = out.transpose(3, 0, 1, 2)

        return out

    def _im2col(self, x):
        w = self.params['w']
        pad, stride = self.pad, self.stride
        _, _, filter_height, filter_width = w.shape

        if filter_height == 1 and filter_width == 1 and pad == 0:
            return x.transpose(1, 2, 3, 0).reshape(x.shape[1], -1)
        else:
            x_cpu = K.get_cpu_array(x)
            return K.asarray(
                im2col(x_cpu, filter_height, filter_width, pad, stride))

    def backward(self, dout, mode):
        w = self.params['w']
        x_cols, num_filters = self.x_cols, w.shape[0]
        N, C, H, W = self.x_shape

        db = K.sum(dout, axis=(0, 2, 3))

        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
        dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

        dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
        dx = self._col2im(dx_cols)

        grads = {
            'w': dw,
            'b': db,
        }

        return dx, grads

    def _col2im(self, dx_cols):
        w, stride, pad = self.params['w'], self.stride, self.pad
        _, _, filter_height, filter_width = w.shape
        N, C, H, W = self.x_shape

        if filter_height == 1 and filter_width == 1 and pad == 0:
            return dx_cols.reshape(C, H, W, N).transpose(3, 0, 1, 2)
        else:
            dx_cols_cpu = K.get_cpu_array(dx_cols)
            return K.asarray(
                col2im(dx_cols_cpu, N, C, H, W, filter_height, filter_width,
                       pad, stride))
