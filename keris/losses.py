import keris.backend as K
from keris.layers import layer

import keris.backend as K
from keris.initializers import initializers


class Loss:
    def __init__(self):
        self.trainable = False
        self.params = {}
        self.name = 'loss'

    def __call__(self, input_layer):
        # Initialize parameters if layer is trainable
        in_shape = input_layer.shape
        if self.trainable:
            self._initialize_params(in_shape)

        return self

    def _initialize_params(self, input_shape):
        raise NotImplementedError

    def _with_initializer(self, name, shape, dtype=K.float32):
        if name not in initializers:
            raise ValueError("initializer does not exists")

        return initializers[name](shape, dtype)

    def update_param(self, key, val):
        self.params[key] = val

    def forward(self, x, mode):
        raise NotImplementedError

    def backward(self, dout, mode):
        raise NotImplementedError


class SoftmaxCrossEntropy(Loss):
    def __init__(self, weighted=False, categorical=True):
        super().__init__()
        self.categorical = categorical
        if weighted:
            self.trainable = True
            self.name = 'softmax_cross_entropy'
        else:
            self.name = 'weighted_softmax_crossentropy'

    def _initialize_params(self, input_shape):
        self.params['w'] = K.ones(input_shape)

    def forward(self, x, y, mode):
        trainable, categorical = self.trainable, self.categorical
        N = x.shape[0]

        if categorical:
            y = y.argmax(axis=1)

        self.x, self.y, self.batch_size = x, y, N

        acc = (x.argmax(axis=1) == y).mean()

        probs = K.exp(x - x.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        if trainable:
            w, self.probs = self.params['w'], probs
            probs *= w
        log = K.log(probs[K.arange(N), y])
        loss = -K.sum(log) / N

        return acc, loss

    def backward(self, mode):
        dx, y, N = self.x.copy(), self.y, self.batch_size
        trainable = self.trainable

        grads = None
        if trainable:
            w = self.params['w']
            dx *= w
            grads = {'w': self.probs}

        dx[K.arange(N), y] -= 1
        dx /= N

        return dx, grads


categorical_softmax_crossentropy = SoftmaxCrossEntropy(categorical=True)
binary_softmax_crossentropy = SoftmaxCrossEntropy(categorical=False)
categorical_weighted_softmax_crossentropy = SoftmaxCrossEntropy(
    weighted=True, categorical=True)
binary_weighted_softmax_crossentropy = SoftmaxCrossEntropy(
    weighted=True, categorical=False)

# def categorical_softmax_crossentropy(x, y):
#     y = y.argmax(axis=1)
#     return binary_softmax_crossentropy(x, y)


# def binary_softmax_crossentropy(x, y):
#     acc = (x.argmax(axis=1) == y).mean()

#     probs = K.exp(x - x.max(axis=1, keepdims=True))
#     probs /= probs.sum(axis=1, keepdims=True)
#     N = x.shape[0]
#     log = K.log(probs[K.arange(N), y])
#     loss = -K.sum(log) / N

#     dx = x.copy()
#     dx[K.arange(N), y] -= 1
#     dx /= N
#     return acc, loss, dx


# def binary_weighted_softmax_crossentropy(x, y):
#     acc = (x.argmax(axis=1) == y).mean()

#     probs = K.exp(x - x.max(axis=1, keepdims=True))
#     probs /= probs.sum(axis=1, keepdims=True)
#     N = x.shape[0]
#     log = K.log(probs[K.arange(N), y])
#     loss = -K.sum(log) / N

#     dx = x.copy()
#     dx[K.arange(N), y] -= 1
#     dx /= N
#     return acc, loss, dx
