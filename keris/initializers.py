import keris.backend as K
from math import sqrt


class Initializer:
    def __call__(self, shape, dtype=K.float32):
        raise NotImplementedError


class RandomNormal(Initializer):
    def __init__(self, mean=0, stddev=1):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape, dtype=K.float32):
        mean, stddev = self.mean, self.stddev

        return K.random.normal(loc=mean, scale=stddev, size=shape).astype(dtype)


class RandomUniform(Initializer):
    def __init__(self, low=-1, high=1):
        self.low = low
        self.high = high

    def __call__(self, shape, dtype=K.float32):
        return K.random.uniform(low=self.low, high=self.high, size=shape)


class GlorotNormal(Initializer):
    def __call__(self, shape, dtype=K.float32):
        if len(shape) == 2:
            n_in, n_out = shape[0], shape[1]
        elif len(shape) == 4:
            filter_wide = shape[2] * shape[3]
            n_in = shape[1] * filter_wide
            n_out = shape[0] * filter_wide
        else:
            raise ValueError("shape must be 2 or 4")

        stddev = sqrt(2 / (n_in + n_out))

        return K.random.normal(loc=0, scale=stddev, size=shape).astype(K.float32)


initializers = {
    "random_normal": RandomNormal(),
    "random_uniform": RandomUniform(),
    "glorot_normal": GlorotNormal()
}
