# -*- coding: utf-8 -*-

from tensorflow.keras.callbacks import Callback

class Save(Callback):
    """
    Save pretrained weights for custom models from utils.models
    path: (str) parent directory path
    monitor: (str) monitor value to add to weights'name
    """
    def __init__(self, path = "./", monitor = 'loss'):
        super(Save, self).__init__()
        self.path = path
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs = None):
        path = f"{self.path}{epoch}-{logs[self.monitor]}"
        self.model.save_pretrained(path)


class prediction_history(Callback):
    def __init__(self):
        self.predhis = []
    def on_epoch_end(self, epoch, logs={}):
        self.predhis.append(model.predict(test_dataset, verbose=1))