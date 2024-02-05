import numpy as np
import os
import glob
import h5py


def load_data(options):
    print("Loading dataset: ")
    all_data = np.array(glob.glob(options["train_data"]))

    if options["masking"]:
        mask_amp_phase = np.load(options["masking"], allow_pickle=True)

        return all_data, mask_amp_phase

    return all_data, None


def split_data(data_size, test_size=0.1):
    N_train = int(data_size * (1 - test_size * 2))
    N_test = int(data_size * test_size)
    N_val = data_size - N_train - N_test

    data_perm = np.random.permutation(data_size)
    train, valid, test, extra = np.split(data_perm, [N_train, N_train + N_val, N_train + N_val + N_test])

    return train, valid, test, extra


def load_exp_data(cfg):
    data = np.load(cfg["hyper"]["train_data"], allow_pickle=True)["arr_0"]
    size = cfg["model"]["img_size"]

    obs = np.sqrt(data[:, 250 - size // 2 : 250 + size // 2, 249 - size // 2 : 249 + size // 2])

    return obs


def load_simu_data(cfg):
    data = np.load(cfg["hyper"]["train_data"], allow_pickle=True)["arr_0"]
    obs = np.random.poisson(data).astype(np.float32)

    return np.sqrt(obs)
    # return data
