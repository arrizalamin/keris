import numpy as np


def get_array_module(x):
    if isinstance(x, np.ndarray):
        return np
    else:
        import cupy
        return cupy


def get_cpu_array(x):
    if isinstance(x, np.ndarray):
        return x
    nd = get_array_module(x)
    return nd.asnumpy(x)
