from keris.callbacks import Callback


class BaseLogger(Callback):
    def on_train_begin(self, logs=None):
        self.history = self.model.metrics.copy()
        self.step = 0

    def on_epoch_end(self, epoch, logs=None):
        self.step += 1
        for key, val in logs.copy().items():
            self.history[key] += val

            self.model.metrics['avg_' + key] = self.history[key] / self.step
