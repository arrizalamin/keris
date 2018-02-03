from keris.callbacks import Callback


class History(Callback):
    def on_train_begin(self, logs=None):
        self.model.history = []

    def on_epoch_end(self, epoch, logs=None):
        self.model.history.append((epoch, logs))
