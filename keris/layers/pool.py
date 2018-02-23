import keris.backend as K
from keris.layers.layer import Layer
from keris.utils.im2col import im2col, col2im
import warnings


class PoolMethod:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError


class MaxPoolReshapeMethod(PoolMethod):
    def forward(self, x):
        """
        A fast implementation of the forward pass for the max pooling layer
        that uses some clever reshaping.

        This can only be used for square pooling regions that tile the input.
        """
        N, C, H, W = x.shape
        pool_height, pool_width = self.pool_size
        stride = self.stride

        out_height = int(H // pool_height)
        out_width = int(W // pool_width)
        x_reshaped = x.reshape(N, C, out_height, pool_height, out_width,
                               pool_width)
        out = x_reshaped.max(axis=3).max(axis=4)

        self.x, self.x_reshaped, self.out = x, x_reshaped, out
        return out

    def backward(self, dout):
        """
        A fast implementation of the backward pass for the max pooling layer
        that uses some clever broadcasting and reshaping.

        This can only be used if the forward pass was computed using
        max_pool_forward_reshape.

        NOTE: If there are multiple argmaxes, this method will assign gradient
        to ALL argmax elements of the input rather than picking one. In this
        case the gradient will actually be incorrect. However this is unlikely
        to occur in practice, so it shouldn't matter much. One possible
        solution is to split the upstream gradient equally among all argmax
        elements; this should result in a valid subgradient. You can make this
        happen by uncommenting the line below; however this results in a
        significant performance penalty (about 40% slower) and is unlikely to
        matter in practice so we don't do it.
        """
        x, x_reshaped, out = self.x, self.x_reshaped, self.out

        dx_reshaped = K.zeros_like(x_reshaped)
        out_newaxis = out[:, :, :, None, :, None]
        mask = (x_reshaped == out_newaxis)
        dout_newaxis = dout[:, :, :, None, :, None]
        dout_broadcast, _ = K.broadcast_arrays(dout_newaxis, dx_reshaped)
        dx_reshaped[mask] = dout_broadcast[mask]
        dx_reshaped /= mask.sum(axis=(3, 5), keepdims=True)
        dx = dx_reshaped.reshape(x.shape)

        return dx


class MaxPoolIm2colMethod(PoolMethod):
    def forward(self, x):
        """
        An implementation of the forward pass for max pooling based on im2col.

        This isn't much faster than the naive version, so it should be avoided
        if possible.
        """
        N, C, H, W = x.shape
        pool_height, pool_width = self.pool_size
        stride = self.stride

        out_height = int((H - pool_height) // stride + 1)
        out_width = int((W - pool_width) // stride + 1)

        x_split = x.reshape(N * C, 1, H, W)
        x_split_cpu = K.get_cpu_array(x_split)
        x_cols = K.asarray(
            im2col(x_split_cpu, pool_height, pool_width, 0, stride))
        x_cols_argmax = x_cols.argmax(axis=0)
        x_cols_max = x_cols[x_cols_argmax, K.arange(x_cols.shape[1])]
        out = x_cols_max.reshape(
            out_height, out_width, N, C).transpose(2, 3, 0, 1)

        self.x, self.x_cols, self.x_cols_argmax = x, x_cols, x_cols_argmax

        return out

    def backward(self, dout):
        """
        An implementation of the backward pass for max pooling based on im2col.

        This isn't much faster than the naive version, so it should be avoided
        if possible.
        """
        x, x_cols, x_cols_argmax = self.x, self.x_cols, self.x_cols_argmax
        N, C, H, W = x.shape
        pool_height, pool_width = self.pool_size
        stride = self.stride

        dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
        dx_cols = K.zeros_like(x_cols)
        dx_cols[x_cols_argmax, K.arange(dx_cols.shape[1])] = dout_reshaped
        dx_cols_cpu = K.get_cpu_array(dx_cols)
        dx = K.asarray(
            col2im(dx_cols_cpu, (N * C), 1, H, W, pool_height, pool_width,
                   0, stride))
        dx = dx.reshape(x.shape)

        return dx


class MaxPooling2D(Layer):
    def __init__(self, name, pool_size=(2, 2), stride=1):
        super().__init__(name)
        self.stride = stride
        self.pool_size = pool_size

    def _get_output_shape(self, input_shape):
        C, H, W = input_shape
        (filter_H, filter_W) = self.pool_size
        stride = self.stride

        H_out = int((H - filter_H) // stride + 1)
        W_out = int((W - filter_W) // stride + 1)

        return (C, H_out, W_out)

    def forward(self, x, mode):
        stride = self.stride
        pool_height, pool_width = self.pool_size
        _, _, H, W = x.shape

        # Check dimensions
        if (H - pool_height) % stride != 0:
            warnings.warn(
                'layer %s: pool incomplete on input width' % self.name,
                UserWarning)
        if (W - pool_width) % stride != 0:
            warnings.warn(
                'layer %s: pool incomplete on input height' % self.name,
                UserWarning)

        same_size = pool_height == pool_width == stride
        tiles = H % pool_height == 0 and W % pool_width == 0
        if same_size and tiles:
            self.method = MaxPoolReshapeMethod(self.pool_size, stride)
        else:
            self.method = MaxPoolIm2colMethod(self.pool_size, stride)

        out = self.method.forward(x)

        return out

    def backward(self, dout, mode):
        dout = self.method.backward(dout)

        return dout, None


class GlobalAveragePooling2D(Layer):
    def _get_output_shape(self, input_shape):
        self.input_shape = input_shape
        self.shape = input_shape[0]
        return self.shape

    def forward(self, x, mode):
        N, C, _, _ = x.shape
        self.N = N
        out = x.mean(axis=(2, 3))
        out.reshape(N, C)
        return out

    def backward(self, dout, mode):
        N = self.N
        C, H, W = self.input_shape
        avg = 1 / (H * W)
        dx = K.full((N, C, H, W), avg, dtype=K.float32)
        for i in range(N):
            for j in range(C):
                dx[i, j, :, :] *= dout[i, j]
        return dx, None
