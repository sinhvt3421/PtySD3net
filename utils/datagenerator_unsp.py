import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from tensorflow.keras.utils import Sequence
from math import ceil
import numpy as np
from ase.db import connect
import numpy as np
from random import shuffle
import tensorflow as tf
import cv2 as cv
from tensorflow.keras.preprocessing.sequence import pad_sequences

RNG_SEED = 123
logger = logging.getLogger(__name__)


class DataIteratorUsp(Sequence):
    """
    Create Data interator over dataset
    """

    def __init__(self, data, batch_size=16, n_time=5, sample=False):
        """_summary_

        Args:
            target (_type_): _description_
            batch_size (int, optional): _description_. Defaults to 32.
            sample (bool, optional): _description_. Defaults to False.
        """
        self.batch_size = batch_size
        self.data = data
        self.n_time = n_time
        self.data_indexes = range(len(self.data) - self.n_time)
        self.sample = sample

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.random.choice(self.data_indexes, len(self.data) // self.n_time)

    def __len__(self):
        return ceil(len(self.data) / (self.batch_size * self.n_time))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        # if self.sample:
        #     diff = [self.data[k : k + self.n_time] for k in indexes]
        #     intensity = np.random.poisson(diff).astype(np.float32)

        #     return np.sqrt(intensity), np.sqrt(intensity) ** 2

        # else:
        diff = np.array([self.data[k : k + self.n_time] for k in indexes])
        return diff, diff**2
