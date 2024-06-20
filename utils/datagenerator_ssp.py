import logging
from tensorflow.keras.utils import Sequence
from math import ceil
import numpy as np
from random import shuffle

RNG_SEED = 2134
logger = logging.getLogger(__name__)


class DataIteratorSsp(Sequence):
    """
    Create Data interator over dataset for self-supervised training
    """

    def __init__(self, data, batch_size=16, n_time=5, image_size=256):
        """_summary_

        Args:
            target (_type_): _description_
            batch_size (int, optional): _description_. Defaults to 32.
            sample (bool, optional): _description_. Defaults to False.
        """
        self.batch_size = batch_size
        self.data = data
        self.n_time = n_time
        self.data_indexes = range(len(self.data) - self.n_time + 1)
        self.image_size = image_size
        self.padding = self.image_size - self.data.shape[-1]

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.random.choice(self.data_indexes, len(self.data) // self.n_time)

    def __len__(self):
        return ceil(len(self.data) / (self.batch_size * self.n_time))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        if self.n_time > 1:
            diff = np.array([self.data[k : k + self.n_time] for k in indexes])
        else:
            diff = self.data[indexes]

        if self.padding > 0:
            pad = [(0, 0)] * diff.ndim
            pad[-1], pad[-2] = (self.padding // 2, self.padding // 2), (self.padding // 2, self.padding // 2)
            diff_p = np.pad(diff, pad)
            return diff_p, diff**2

        return diff, diff**2
