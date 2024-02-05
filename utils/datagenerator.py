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


class DataIterator(Sequence):
    """
    Create Data interator over dataset
    """

    def __init__(self, data_path, target, batch_size=32, shuffle=False, mask=None, mode="2d"):
        """_summary_

        Args:
            data_path (_type_): _description_
            target (_type_): _description_
            batch_size (int, optional): _description_. Defaults to 32.
            shuffle (bool, optional): _description_. Defaults to False.
            mask (_type_, optional): _description_. Defaults to None.
        """
        self.batch_size = batch_size

        if target == "a_p":
            self.targets = ["Amp", "Phase"]
        elif target == "a":
            self.targets = ["Amp"]
        else:
            self.targets = ["Phase"]

        self.data_path = data_path
        self.shuffle = shuffle
        self.mode = mode
        self.mask_amp_phase = None

        if mask is not None:
            self.mask_amp_phase = mask

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_path))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return ceil(len(self.data_path) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_data = [np.load(self.data_path[x], allow_pickle=True)["arr_0"] for x in indexes]

        data_diff = [t[-1] for t in batch_data]

        data_diff = np.random.poisson(np.stack(data_diff, 0)).astype("float32")

        y = [t[:2] for t in batch_data]

        y = np.stack(y, 0)

        if self.mask_amp_phase is not None:
            y = y * self.mask_amp_phase[None, ...]
        a = y[:, 0]

        p = y[:, 1]
        return np.sqrt(data_diff), (a, p, a, p, data_diff, data_diff)
