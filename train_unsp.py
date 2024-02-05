import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")

tf.config.experimental.set_memory_growth(physical_devices[0], True)

import yaml
import numpy as np

# from ptycle_net.ptycle.cyclegan import CycleGAN
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import *

from ptynet.models import PtySPINet
from ptynet.losses import *
from utils.general import load_exp_data, load_simu_data
from utils.datagenerator_unsp import DataIteratorUsp
import random
import argparse
import time


def set_seed(seed=2134):
    # tf.keras.utils.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    # os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    # os.environ["TF_DETERMINISTIC_OPS"] = "1"
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def main(args):
    set_seed(0)

    config = yaml.safe_load(open(args.dataset))

    unsp_model = PtySPINet(config, args.pretrained, mode="train")
    unsp_model.model.summary()

    data_exp = load_simu_data(config) if config["hyper"]["sample"] else load_exp_data(config)
    trainIter = DataIteratorUsp(data_exp, batch_size=config["hyper"]["batch_size"], sample=config["hyper"]["sample"])

    lr_schedule = CosineDecay(
        1e-3,
        1.0 * 50 * len(trainIter),
        alpha=0.5,
        name=None,
    )
    loss = negative_log_loss if config["hyper"]["dist"] else r_f_I

    unsp_model.model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
    )
    if not os.path.exists("{}/models/".format(config["hyper"]["save_path"])):
        os.makedirs("{}/models/".format(config["hyper"]["save_path"]))

    callbacks = []

    callbacks.append(
        ModelCheckpoint(
            filepath="{}/models/model_unsp.tf".format(
                config["hyper"]["save_path"],
            ),
            monitor="loss",
            save_weights_only=True,
            verbose=2,
            save_best_only=True,
        )
    )

    yaml.safe_dump(
        config,
        open(
            "{}/config.yaml".format(config["hyper"]["save_path"]),
            "w",
        ),
        default_flow_style=False,
    )
    t = time.time()
    hist = unsp_model.model.fit(
        trainIter,
        epochs=50,
        callbacks=callbacks,
        verbose=1,
        shuffle=False,
        use_multiprocessing=True,
        workers=4,
    )
    print("Total training time: ", time.time() - t)

    tf.keras.backend.clear_session()
    infer_model = PtySPINet(config, mode="infer")
    infer_model.model.load_weights("{}/models/model_unsp.tf".format(config["hyper"]["save_path"])).expect_partial()

    all_predict_a = []
    all_predict_p = []

    t = time.time()
    for idx in range(0, len(data_exp) - 4, 5):
        # for idx in range(0, 50, 5):
        a, p = infer_model.model(data_exp[idx : idx + 5][None, ...])

        all_predict_a.append(a)
        all_predict_p.append(p)

    print("Total Inferences time: ", time.time() - t)

    all_predict_a = np.array(all_predict_a).reshape(-1, a.shape[-1], a.shape[-1])
    all_predict_p = np.array(all_predict_p).reshape(-1, a.shape[-1], a.shape[-1])
    np.savez_compressed(
        "{}/object_reconstruction.npz".format(config["hyper"]["save_path"]), [all_predict_a, all_predict_p]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("dataset", type=str, help="Path to dataset configs")

    parser.add_argument("--pretrained", type=str, default="", help="Path to pretrained model (optional)")

    args = parser.parse_args()
    main(args)
