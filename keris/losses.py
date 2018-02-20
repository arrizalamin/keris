import keris.backend as K


def categorical_softmax_crossentropy(x, y):
    y = y.argmax(axis=1)
    return binary_softmax_crossentropy(x, y)


def binary_softmax_crossentropy(x, y):
    acc = (x.argmax(axis=1) == y).mean()

    probs = K.exp(x - x.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    N = x.shape[0]
    log = K.log(probs[K.arange(N), y])
    loss = -K.sum(log) / N

    dx = x.copy()
    dx[K.arange(N), y] -= 1
    dx /= N
    return acc, loss, dx
