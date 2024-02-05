import glob
import os

import numpy as np
import tensorflow as tf
import yaml
from ptynet.layers import *
from ptynet.losses import *
from tensorflow.keras import Model
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import Conv2D, Input, Lambda
from tensorflow.keras.optimizers.schedules import CosineDecay, ExponentialDecay
from utils.datagenerator import DataIterator
from utils.general import load_data, split_data


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
        all_data, mask = load_data(cfgh)

        self.config["hyper"]["data_size"] = len(all_data)
        train, valid, test, extra = split_data(len(all_data), test_size=cfgh["test_size"])

        assert len(extra) == 0, "Split was inexact {} {} {} {}".format(len(train), len(valid), len(test), len(extra))

        print(
            "Number of train data : ",
            len(train),
            " , Number of valid data: ",
            len(valid),
            " , Number of test data: ",
            len(test),
        )

        self.trainIter, self.validIter, self.testIter = [
            DataIterator(
                batch_size=cfgh["batch_size"],
                data_path=all_data[indices],
                target=cfgh["target"],
                shuffle=(len(indices) == len(train)),
                mask=mask,
                mode=cfgh["mode"],
            )
            for indices in (train, valid, test)
        ]

    def create_callbacks(self):
        callbacks = []
        callbacks.append(
            ModelCheckpoint(
                filepath="{}_{}/models/model_{}.h5".format(
                    self.config["hyper"]["save_path"],
                    self.config["hyper"]["target"],
                    self.config["hyper"]["target"],
                ),
                monitor="loss",
                save_weights_only=True,
                verbose=2,
                save_best_only=True,
            )
        )

        callbacks.append(EarlyStopping(monitor="loss", patience=100))

        callbacks.append(LearningRateTracker())

        return callbacks

    def create_loss_op(self):
        loss_name = self.config["hyper"]["loss"]
        if loss_name == "mse":
            loss = masked_MSEloss
        elif loss_name == "psnr":
            loss = PSNRLoss(self.config["hyper"]["max_val"])
        elif loss_name == "ssim":
            loss = SSIMLoss(self.config["hyper"]["max_val"])
        else:
            loss = "mae"

        metrics = [
            # SSIMMetric(self.config["hyper"]["max_val"]),
            # PSNRMetric(self.config["hyper"]["max_val"]),
        ]

        loss_weights = None
        if self.config["hyper"]["target"] == "a_p":
            loss = {"amplitude": masked_MSEloss_v(1.0), "phase": masked_MSEloss_v(0.0)}
            loss_weights = {"amplitude": 10.0, "phase": 1.0}

        if self.config["hyper"]["target"] == "a_p_i":
            loss = {
                "amplitude": SEloss,
                "phase": SEloss,
                "intensity_sample": negative_log_loss,
                "intensity_sample_r": negative_log_loss,
                "amplitude_r": SEloss,
                "phase_r": SEloss,
            }
            loss_weights = {
                "amplitude": 10.0,
                "phase": 10.0,
                "amplitude_r": 1.0,
                "phase_r": 1.0,
                "intensity_sample_r": 1.0,
                "intensity_sample": 1.0,
            }

        return loss, loss_weights, metrics

    def train(self, epochs):
        lr_schedule = CosineDecay(
            self.config["hyper"]["lr"],
            1.0 * len(self.trainIter) * epochs,
            alpha=0.2,
            name=None,
        )

        loss, loss_weights, metrics = self.create_loss_op()

        self.model.compile(
            loss=loss,
            loss_weights=loss_weights,
            optimizer=tf.keras.optimizers.Adam(lr_schedule),
            metrics=metrics,
        )

        if not os.path.exists(
            "{}_{}/models/".format(self.config["hyper"]["save_path"], self.config["hyper"]["target"])
        ):
            os.makedirs("{}_{}/models/".format(self.config["hyper"]["save_path"], self.config["hyper"]["target"]))

        callbacks = self.create_callbacks()

        yaml.safe_dump(
            self.config,
            open(
                "{}_{}/config.yaml".format(self.config["hyper"]["save_path"], self.config["hyper"]["target"]),
                "w",
            ),
            default_flow_style=False,
        )

        self.hist = self.model.fit(
            self.trainIter,
            epochs=epochs,
            validation_data=self.validIter,
            callbacks=callbacks,
            verbose=1,
            shuffle=False,
            use_multiprocessing=True,
            workers=4,
        )
        return self.hist
