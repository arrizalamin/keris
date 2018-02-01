from keris.callbacks import Callback
from time import time


class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', save_best_only=False):
        if monitor not in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            raise ValueError('metric to monitor is not available')
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only

        self.best = None
        self.mode = 'max' if monitor in ['train_acc', 'val_acc'] else 'min'

    def on_epoch_end(self, epoch, logs=None):
        if not self.save_best_only:
            self._save_model(epoch)

        best, mode = self.best, self.mode
        metric = logs[self.monitor]
        if best is None:
            self.best = metric
            self._save_model(epoch)
            return
        is_best = metric > best if mode == 'max' else best > metric
        if is_best:
            self.best = metric
            self._save_model(epoch)

    def _save_model(self, epoch):
        filepath = '%s-%d-%f' % (self.filepath, epoch, time())
        self.model.save(filepath)
