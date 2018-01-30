import numpy as np
from math import ceil
from trainer import Trainer


class BatchGenerator:
    def __init__(self, train_generator, val_generator, train_steps, val_steps):
        self.train_generator = train_generator
        self.val_generator = val_generator

        self.train_steps = train_steps
        self.val_steps = val_steps

    def get_batch(self, data, step):
        if data == 'train':
            return next(self.train_generator)
        elif data == 'validation':
            return next(self.val_generator)
        else:
            raise ValueError('data must be train or validation')


class BatchLargeData:
    def __init__(self, train_data, val_data, batch_size):
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size

        self.train_steps = ceil(len(train_data) / batch_size)
        self.val_steps = ceil(len(val_data) / batch_size)

    def get_batch(self, data, step):
        batch_size = self.batch_size
        if data == 'train':
            x_batch, y_batch = zip(
                *self.train_data[step * batch_size: (step + 1) * batch_size])
            return np.array(x_batch), np.array(y_batch)
        elif data == 'validation':
            x_batch, y_batch = zip(
                *self.val_data[step * batch_size: (step + 1) * batch_size])
            return np.array(x_batch), np.array(y_batch)
        else:
            raise ValueError('data must be train or validation')


class Model(Trainer):
    def fit(self, train_data, val_data, epochs=10, batch_size=32,
            checkpoint=None, callback=None):
        batch = BatchLargeData(train_data, val_data, batch_size=batch_size)
        self._fit(batch, epochs=epochs,
                  checkpoint=checkpoint, callback=callback)

    def fit_generator(self, train_generator, val_generator, train_steps,
                      val_steps, epochs=10, checkpoint=None, callback=None):
        batch = BatchGenerator(train_generator, val_generator,
                               train_steps=train_steps, val_steps=val_steps)
        self._fit(batch, epochs=epochs,
                  checkpoint=checkpoint, callback=callback)

    def _get_parameters(self):
        params = {}
        for layer in self.layers:
            if layer.trainable:
                params[layer.name] = layer.params

        return params

    def save(self, filename):
        params = self._get_parameters()
        filename += '.npy'
        np.save(filename, params)

    def load(self, filename):
        filename += '.npy'
        params = np.load(filename).item()
        for layer in self.layers:
            if not (layer.trainable or layer.name in params):
                continue
            next_params = params[layer.name]
            for name, param in next_params.items():
                if layer.params[name].shape != next_params[name].shape:
                    raise Exception(
                        'layer %s\'s parameters have different shape than saved parameters' % layer.name)
            layer.params = next_params
