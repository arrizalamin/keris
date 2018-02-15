import numpy as np


def categorical_softmax_crossentropy(x, y):
    y = np.argmax(y, axis=1)
    return binary_softmax_crossentropy(x, y)


def binary_softmax_crossentropy(x, y):
    acc = (np.argmax(x, axis=1) == y).mean()

    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    log = np.log(probs[np.arange(N), y])
    loss = -np.sum(log) / N

    dx = x.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return acc, loss, dx
