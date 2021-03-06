import keris.backend as K
import numpy as np
from math import ceil
from tqdm import trange
from keris.trainer import Trainer
from keris.callbacks import BaseLogger, History


class BatchGenerator:
    def __init__(self, train_generator, val_generator, train_steps, val_steps,
                 data_format='channels_last'):
        self.train_generator = train_generator
        self.val_generator = val_generator

        self.train_steps = train_steps
        self.val_steps = val_steps
        self.data_format = data_format

    def get_batch(self, data, step):
        if data == 'train':
            x_batch, y_batch = next(self.train_generator)
            if self.data_format == 'channels_last':
                x_batch = K.asarray(x_batch.transpose(0, 3, 1, 2),
                                    dtype=K.float32)
                y_batch = K.asarray(y_batch)
                return x_batch, y_batch
        elif data == 'validation':
            x_batch, y_batch = next(self.val_generator)
            if self.data_format == 'channels_last':
                x_batch = K.asarray(x_batch.transpose(0, 3, 1, 2),
                                    dtype=K.float32)
                y_batch = K.asarray(y_batch)
                return x_batch, y_batch
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
            return K.asarray(x_batch, dtype=K.float32), K.asarray(y_batch)
        elif data == 'validation':
            x_batch, y_batch = zip(
                *self.val_data[step * batch_size: (step + 1) * batch_size])
            return K.asarray(x_batch, dtype=K.float32), K.asarray(y_batch)
        else:
            raise ValueError('data must be train or validation')


class Model(Trainer):
    def _fit(self, batch, epochs=10, checkpoint=None, callbacks=[]):
        self.stop_training = False

        default_callbacks = [BaseLogger(), History()]
        self._init_callbacks(callbacks, with_default=default_callbacks)

        self._reset_metrics()

        te = trange(epochs)
        train_steps = batch.train_steps
        val_steps = batch.val_steps
        self._call_callbacks('on_train_begin')
        for epoch in te:
            if self.stop_training:
                break

            self._call_callbacks('on_epoch_begin')

            t_loss, t_acc, v_loss, v_acc = 0, 0, 0, 0

            ts = trange(train_steps)
            for t in ts:
                x_batch, y_batch = batch.get_batch('train', t)
                x_batch = (x_batch / 127.5) - 1
                step_acc, step_loss = self.train_on_batch(x_batch, y_batch)
                ts.set_description('train: loss=%g, acc=%g' %
                                   (step_loss, step_acc))
                t_loss += step_loss
                t_acc += step_acc
            self.metrics['train_loss'] = t_loss / train_steps
            self.metrics['train_acc'] = t_acc / train_steps

            for t in range(val_steps):
                x_batch, y_batch = batch.get_batch('validation', t)
                x_batch = (x_batch / 127.5) - 1

                val_loss, val_acc, _ = self._forward(
                    x_batch, y_batch, mode='test')
                v_acc += val_acc
                v_loss += val_loss
            self.metrics['val_loss'] = v_loss / val_steps
            self.metrics['val_acc'] = v_acc / val_steps
            te.set_description('loss: %g acc:%g' % (
                self.metrics['val_loss'], self.metrics['val_acc']))

            self._call_callbacks('on_epoch_end')

            self.epoch += 1
            self.optimizer.decrease_lr()
        self._call_callbacks('on_train_end')

        # print new line to prevent next prompt override progress bar
        print()

    def _init_callbacks(self, callbacks, with_default=[]):
        self.callbacks = with_default
        self.callbacks.extend(callbacks)
        for callback in self.callbacks:
            callback.set_model(self)

    def _call_callbacks(self, method_name):
        for callback in self.callbacks:
            if method_name in ['on_train_begin', 'on_train_end']:
                getattr(callback, method_name)(self.metrics)
            else:
                getattr(callback, method_name)(self.epoch, self.metrics)

    def _reset_metrics(self):
        metrics_name = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
        self.metrics = {key: 0 for key in metrics_name}

    def fit(self, train_data, val_data, epochs=10, batch_size=32,
            checkpoint=None, callbacks=[]):
        batch = BatchLargeData(train_data, val_data, batch_size=batch_size)
        self._fit(batch, epochs=epochs,
                  checkpoint=checkpoint, callbacks=callbacks)

    def fit_generator(self, train_generator, val_generator, train_steps,
                      val_steps, epochs=10, checkpoint=None, callbacks=[]):
        batch = BatchGenerator(train_generator, val_generator,
                               train_steps=train_steps, val_steps=val_steps)
        self._fit(batch, epochs=epochs,
                  checkpoint=checkpoint, callbacks=callbacks)

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
