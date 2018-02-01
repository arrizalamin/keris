from keris.callbacks import Callback


class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss', min_delta=0, patience=3):
        if monitor not in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            raise ValueError('metric to monitor is not available')
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience

        self.best = None
        self.patience_counter = 0
        self.mode = 'max' if monitor in ['train_acc', 'val_acc'] else 'min'

    def on_epoch_end(self, epoch, logs=None):
        best, mode, min_delta = self.best, self.mode, self.min_delta
        metric = logs[self.monitor]
        if best is None:
            delta = metric
        else:
            delta = metric - best if mode == 'max' else best - metric
        if delta >= min_delta:
            self.best = metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.model.stop_training = True
