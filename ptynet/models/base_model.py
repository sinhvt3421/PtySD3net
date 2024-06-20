import os

import numpy as np
import tensorflow as tf
import yaml
from ptynet.layers import *
from ptynet.losses import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers.schedules import CosineDecay, ExponentialDecay
from utils.datagenerator_ssp import DataIteratorSsp
from utils.general import dataset_functions
import time


class LearningRateTracker(tf.keras.callbacks.Callback):
    def on_epoch_end(self, e, log):
        optimizer = self.model.optimizer
        print("\nLR: {:.6f}\n".format(optimizer._decayed_lr(tf.float32)))


class PtyBase:
    def __init__(self, config, model, pretrained=""):
        self.config = config
        self.model = model

    def create_dataset(self):
        cfgh = self.config["hyper"]
        self.data_exp = dataset_functions[cfgh["sample"]](self.config)

        self.trainIter = DataIteratorSsp(
            self.data_exp,
            batch_size=cfgh["batch_size"],
            image_size=self.config["model"]["img_size"],
            n_time=(
                1
                if (self.config["model"]["mode"] == "2d") or (self.config["model"]["mode"] == "ptychonn")
                else self.config["hyper"]["n_time"]
            ),
        )

    def create_callbacks(self):
        callbacks = []
        callbacks.append(
            ModelCheckpoint(
                filepath="{}/models/model_unsp.tf".format(
                    self.config["hyper"]["save_path"],
                ),
                monitor="loss",
                save_weights_only=True,
                verbose=2,
                save_best_only=True,
            )
        )
        callbacks.append(LearningRateTracker())
        return callbacks

    def create_loss_op(self):
        loss = negative_log_loss(self.config["hyper"]['loss']) if self.config["hyper"]["dist"] else masked_SEloss
        return loss

    def train(self, epochs):
        lr_schedule = CosineDecay(
            self.config["hyper"]["lr"],
            1.0 * epochs * len(self.trainIter),
            alpha=0.2,
            name=None,
        )
        loss = self.create_loss_op()

        self.model.compile(
            # No loss for prediction of amplitude and phase information
            loss=[loss, None, None],
            optimizer=tf.keras.optimizers.Adam(lr_schedule),
        )

        if not os.path.exists("{}/models/".format(self.config["hyper"]["save_path"])):
            os.makedirs("{}/models/".format(self.config["hyper"]["save_path"]))

        callbacks = self.create_callbacks()

        yaml.safe_dump(
            self.config,
            open(
                "{}/config.yaml".format(self.config["hyper"]["save_path"]),
                "w",
            ),
            default_flow_style=False,
        )

        self.hist = self.model.fit(
            self.trainIter,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            shuffle=False,
            use_multiprocessing=True,
            workers=4,
        )
        return self.hist

    def inference(self):
        self.model.load_weights("{}/models/model_unsp.tf".format(self.config["hyper"]["save_path"])).expect_partial()

        all_predict_a = []
        all_predict_p = []

        t = time.time()
        padding = self.config["model"]["img_size"] - self.data_exp.shape[-1]

        if padding > 0:
            if self.config["model"]["mode"] == "2d" or self.config["model"]["mode"] == "ptychonn":
                pad = ((0, 0), (padding // 2, padding // 2), (padding // 2, padding // 2))
            else:
                pad = ((0, 0), (0, 0), (padding // 2, padding // 2), (padding // 2, padding // 2))

        if self.config["model"]["mode"] == "2d" or self.config["model"]["mode"] == "ptychonn":
            size = self.config["hyper"]["batch_size"]
        else:
            size = self.config["hyper"]["n_time"]

        for idx in range(0, len(self.data_exp)//size + 1, 1):
            diff = self.data_exp[idx*size : (idx+1)* size]

            if (self.config["model"]["mode"] != "2d") and (self.config["model"]["mode"] != "ptychonn"):
                diff = diff[None, ...]
            if padding > 0:
                diff = np.pad(diff, pad)
            _, a, p = self.model(diff)

            all_predict_a.append(a)
            all_predict_p.append(p)

        print("Total Inferences time: ", time.time() - t)

        all_predict_a = np.array(all_predict_a).reshape(-1, a.shape[-1], a.shape[-1])
        all_predict_p = np.array(all_predict_p).reshape(-1, a.shape[-1], a.shape[-1])
        np.savez_compressed(
            "{}/object_reconstruction_{}.npz".format(self.config["hyper"]["save_path"], self.config["model"]["mode"]),
            [all_predict_a, all_predict_p],
        )

