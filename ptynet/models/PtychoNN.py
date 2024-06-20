import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda, Conv3D, Conv2D, UpSampling2D, MaxPool2D
from ptynet.layers import *
from tensorflow.keras.callbacks import *
from ptynet.models import PtyBase
from ptynet.losses import total_var_3d, total_var
from tensorflow.signal import fft2d, fftshift, ifftshift, ifft2d
import math

import numpy as np


class PtychoNN(PtyBase):
    def __init__(self, config, pretrained=""):
        model = create_model(config)
        if pretrained:
            print("Load pretrained model from ", pretrained)
            model.load_weights(pretrained).expect_partial()
        super(PtychoNN, self).__init__(config=config, model=model)

### All the code from https://github.com/mcherukara/PtychoNN/blob/master/TF2/keras_helper.py 

def Conv_Pool_block(x0, nfilters, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last"):
    x0 = Conv2D(nfilters, (w1, w2), activation="relu", padding=padding, data_format=data_format)(x0)
    x0 = Conv2D(nfilters, (w1, w2), activation="relu", padding=padding, data_format=data_format)(x0)
    x0 = MaxPool2D((p1, p2), padding=padding, data_format=data_format)(x0)
    return x0


def Conv_Up_block(x0, nfilters, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last"):
    x0 = Conv2D(nfilters, (w1, w2), activation="relu", padding=padding, data_format=data_format)(x0)
    x0 = Conv2D(nfilters, (w1, w2), activation="relu", padding=padding, data_format=data_format)(x0)
    x0 = UpSampling2D((p1, p2), data_format=data_format)(x0)
    return x0


def ff_propagation(probe, objects, probe_mode="multi_c"):
    fftconst = probe.shape[-1]
    if "single" in probe_mode:
        pre_exit = probe * objects

        dif = fftshift(fft2d(pre_exit), axes=(-2, -1)) / fftconst
        intensity = tf.abs(dif)
    else:
        pre_exit = probe * tf.expand_dims(objects, 1)

        dif = fftshift(fft2d(pre_exit), axes=(-2, -1)) / fftconst

        intensity = tf.sqrt(tf.reduce_sum(tf.abs(dif) ** 2, 1))
    return intensity**2


def create_model(config):
    cfgm = config["model"]
    cfgh = config["hyper"]

    probs = np.load(cfgh["probe"], allow_pickle=True)

    if cfgh["probe_norm"]:
        # Scale probe amplitude base on exposure time, current 1s normalized
        probs = tf.constant(probs * np.sqrt(float(cfgh["probe_norm"])), dtype="complex64")

    if cfgh["masking"]:
        # Masking probe position
        mask = np.load(cfgh["masking"], allow_pickle=True)[None, :, :].astype(np.float32)
        mask = tf.constant(mask)

    input_img = Input(shape=(cfgm["img_size"], cfgm["img_size"], 1))

    ### All the code from https://github.com/mcherukara/PtychoNN/blob/master/TF2/tf2_train_network.ipynb
    
    x = Conv_Pool_block(input_img, 32, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last")
    x = Conv_Pool_block(x, 64, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last")
    x = Conv_Pool_block(x, 128, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last")
    # Activations are all ReLu

    encoded = x

    # Decoding arm 1
    x1 = Conv_Up_block(encoded, 128, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last")
    x1 = Conv_Up_block(x1, 64, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last")
    x1 = Conv_Up_block(x1, 32, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last")

    decoded1 = Conv2D(1, (3, 3), padding="same", activation="sigmoid")(x1)

    # Decoding arm 2
    x2 = Conv_Up_block(encoded, 128, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last")
    x2 = Conv_Up_block(x2, 64, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last")
    x2 = Conv_Up_block(x2, 32, w1=3, w2=3, p1=2, p2=2, padding="same", data_format="channels_last")

    decoded2 = Conv2D(1, (3, 3), padding="same")(x2)
    decoded2 = Mpi()(decoded2)
    # decoded2 = Lambda(lambda x: math.pi * tf.tanh(x))(decoded2)

    ### 
    
    ### Adding forward FFT for self-supervised learning 

    # Cropping objects, diffraction to match probs shape
    padding = input_img.shape[-2] - probs.shape[-1]
    if padding > 0:
        decoded1 = decoded1[:, padding // 2 : -padding // 2, padding // 2 : -padding // 2]
        decoded2 = decoded2[:, padding // 2 : -padding // 2, padding // 2 : -padding // 2]

    if cfgh["masking"]:
        decoded1 = Lambda(lambda x: tf.squeeze(x, -1) * mask, name="amnp")(decoded1)
        decoded2 = Lambda(lambda x: tf.squeeze(x, -1) * mask, name="phi")(decoded2)

    # forward propagation
    obj = CombineComplex()(decoded1, decoded2)

    # probe function propagation to get the diff
    Psi = Lambda(lambda x: ff_propagation(x[0], x[1]), name="diff_intensity")([probs, obj])

    # Put together
    autoencoder = Model(input_img, [Psi, decoded1, decoded2])
    return autoencoder
