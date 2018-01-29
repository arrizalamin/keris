import numpy as np


def softmax_crossentropy(x, y):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    log = np.log(probs[np.arange(N), y])
    loss = -np.sum(log) / N

    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
