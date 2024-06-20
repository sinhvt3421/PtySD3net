# Keras modules
from tensorflow.keras.layers import Conv3D, MaxPool3D, UpSampling3D, ZeroPadding3D
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Activation
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.activations import sigmoid, tanh, relu
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K
import math
from ptynet.layers import *
from tensorflow.signal import fft2d, fftshift, ifftshift, ifft2d
from ptynet.models import PtyBase

import numpy as np


class AutoPhaseNN(PtyBase):
    def __init__(self, config, pretrained=""):
        model = create_model(config)
        if pretrained:
            print("Load pretrained model from ", pretrained)
            model.load_weights(pretrained).expect_partial()
        super(AutoPhaseNN, self).__init__(config=config, model=model)


#All the code from https://github.com/YudongYao/AutoPhaseNN/blob/main/TF2/keras_helper.py

# encoder layers
def Conv_Pool_block(
    x0, nfilters, w1=3, w2=3, w3=3, p1=1, p2=2, p3=2, Lalpha=0.05, padding="same", data_format="channels_last"
):
    x0 = Conv3D(nfilters, (w1, w2, w3), padding=padding, data_format=data_format)(x0)
    x0 = LeakyReLU(alpha=Lalpha)(x0)
    x0 = BatchNormalization()(x0)

    x0 = Conv3D(nfilters, (w1, w2, w3), padding=padding, data_format=data_format)(x0)
    x0 = LeakyReLU(alpha=Lalpha)(x0)
    x0 = BatchNormalization()(x0)

    x0 = MaxPool3D((p1, p2, p3), padding=padding, data_format=data_format)(x0)
    return x0


def Conv_Pool_block_last(
    x0, nfilters, w1=3, w2=3, w3=3, p1=1, p2=2, p3=2, Lalpha=0.05, padding="same", data_format="channels_last"
):
    x0 = Conv3D(nfilters, (w1, w2, w3), padding=padding, data_format=data_format)(x0)
    x0 = LeakyReLU(alpha=Lalpha)(x0)
    x0 = BatchNormalization()(x0)

    x0 = Conv3D(nfilters, (w1, w2, w3), padding=padding, data_format=data_format)(x0)
    x0 = LeakyReLU(alpha=Lalpha)(x0)
    x0 = BatchNormalization()(x0)

    return x0


# decoder layer
def Conv_Upfirst_block(
    x0, nfilters, w1=3, w2=3, w3=3, p1=1, p2=2, p3=2, Lalpha=0.05, padding="same", data_format="channels_last"
):
    x0 = UpSampling3D((p1, p2, p3), data_format=data_format)(x0)

    x0 = Conv3D(nfilters, (w1, w2, w3), padding=padding, data_format=data_format)(x0)
    x0 = LeakyReLU(alpha=Lalpha)(x0)
    x0 = BatchNormalization()(x0)

    x0 = Conv3D(nfilters, (w1, w2, w3), padding=padding, data_format=data_format)(x0)
    x0 = LeakyReLU(alpha=Lalpha)(x0)
    x0 = BatchNormalization()(x0)

    return x0


def Conv_Upfirst_block_last(x0, nfilters, w1=3, w2=3, w3=3, psize=16, padding="same", data_format="channels_last"):
    # x0 = ZeroPadding3D(padding=psize, data_format=data_format)(x0)

    x0 = Conv3D(nfilters, (w1, w2, w3), activation="swish", padding=padding, data_format=data_format)(x0)
    x0 = BatchNormalization()(x0)

    x0 = Conv3D(nfilters, (w1, w2, w3), activation="swish", padding=padding, data_format=data_format)(x0)
    x0 = BatchNormalization()(x0)

    return x0


def Conv_Upfirst_block_mlast(x0, nfilters, w1=3, w2=3, w3=3, psize=16, padding="same", data_format="channels_last"):
    # x0 = ZeroPadding3D(padding=psize, data_format=data_format)(x0)

    x0 = Conv3D(nfilters, (w1, w2, w3), activation="swish", padding=padding, data_format=data_format)(x0)
    x0 = Conv3D(nfilters, (w1, w2, w3), activation="swish", padding=padding, data_format=data_format)(x0)

    return x0

###

def get_mask(input):

    mask = tf.where(input >= 0.1, tf.ones_like(input), tf.zeros_like(input))
    return mask


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
    fnum = cfgm["filters"]

    probs = np.load(cfgh["probe"], allow_pickle=True)

    if cfgh["probe_norm"]:
        # Scale probe amplitude base on exposure time, current 1s normalized
        probs = tf.constant(probs * np.sqrt(float(cfgh["probe_norm"])), dtype="complex64")

    if cfgh["masking"]:
        # Masking probe position
        mask = np.load(cfgh["masking"], allow_pickle=True)[None, ...]
        mask = tf.constant(mask, dtype="float32")

    input_img = Input(shape=(None, cfgm["img_size"], cfgm["img_size"], 1), dtype="float32")

    ### All the code from https://github.com/YudongYao/AutoPhaseNN/blob/main/TF2/train_network_unsup_3D.py

    # Encoding layers
    # Activations are all leakyReLu
    x = Conv_Pool_block(
        input_img, fnum, w1=3, w2=3, w3=3, p1=1, p2=2, p3=2, Lalpha=0.01, padding="same", data_format="channels_last"
    )
    x = Conv_Pool_block(
        x, fnum * 2, w1=3, w2=3, w3=3, p1=1, p2=2, p3=2, Lalpha=0.01, padding="same", data_format="channels_last"
    )
    x = Conv_Pool_block(
        x, fnum * 4, w1=3, w2=3, w3=3, p1=1, p2=2, p3=2, Lalpha=0.01, padding="same", data_format="channels_last"
    )
    x = Conv_Pool_block(
        x, fnum * 8, w1=3, w2=3, w3=3, p1=1, p2=2, p3=2, Lalpha=0.01, padding="same", data_format="channels_last"
    )
    x = Conv_Pool_block_last(
        x, fnum * 16, w1=3, w2=3, w3=3, p1=1, p2=2, p3=2, Lalpha=0.01, padding="same", data_format="channels_last"
    )

    encoded = x

    # Decoding arm 1
    x1 = Conv_Upfirst_block(
        encoded, fnum * 8, w1=3, w2=3, w3=3, p1=1, p2=2, p3=2, Lalpha=0.01, padding="same", data_format="channels_last"
    )
    x1 = Conv_Upfirst_block(
        x1, fnum * 4, w1=3, w2=3, w3=3, p1=1, p2=2, p3=2, Lalpha=0.01, padding="same", data_format="channels_last"
    )
    x1 = Conv_Upfirst_block(
        x1, fnum * 2, w1=3, w2=3, w3=3, p1=1, p2=2, p3=2, Lalpha=0.01, padding="same", data_format="channels_last"
    )
    x1 = Conv_Upfirst_block(
        x1, fnum * 1, w1=3, w2=3, w3=3, p1=1, p2=2, p3=2, Lalpha=0.01, padding="same", data_format="channels_last"
    )
    x1 = Conv_Upfirst_block_last(x1, fnum * 1, w1=3, w2=3, w3=3, psize=16, padding="same", data_format="channels_last")

    # Decoding arm 2
    x2 = Conv_Upfirst_block(
        encoded, fnum * 4, w1=3, w2=3, p1=1, p2=2, Lalpha=0.01, padding="same", data_format="channels_last"
    )
    x2 = Conv_Upfirst_block(
        x2, fnum * 4, w1=3, w2=3, p1=1, p2=2, Lalpha=0.01, padding="same", data_format="channels_last"
    )
    x2 = Conv_Upfirst_block(
        x2, fnum * 2, w1=3, w2=3, p1=1, p2=2, Lalpha=0.01, padding="same", data_format="channels_last"
    )
    x2 = Conv_Upfirst_block(
        x2, fnum * 1, w1=3, w2=3, p1=1, p2=2, Lalpha=0.01, padding="same", data_format="channels_last"
    )
    x2 = Conv_Upfirst_block_last(x2, fnum * 1, w1=3, w2=3, w3=3, psize=16, padding="same", data_format="channels_last")

    decoded1 = Conv3D(1, (3, 3, 3), padding="same")(x1)
    decoded1 = Lambda(lambda x: sigmoid(tf.squeeze(x, -1)))(decoded1)

    # decoded1 = amp_constraint(name = 'amp')(decoded1)
    # decoded1 = Lambda(lambda x: amp_constraint(x), name='amp')(decoded1)
    # support = Lambda(lambda x: get_mask(x), name="support")(decoded1)

    decoded2 = Conv3D(1, (3, 3, 3), padding="same")(x2)
    # decoded2 = phi_constraint(name='phi')(decoded2)
    # decoded2 = Lambda(lambda x: math.pi * tanh(tf.squeeze(x, -1)))(decoded2)

    decoded2 = Mpi()(decoded2)
    decoded2 = Lambda(lambda x: tf.squeeze(x, -1))(decoded2)

    ### 
    
    ### Adding forward FFT for self-supervised learning 
    
    # Cropping objects, diffraction to match probs shape
    padding = input_img.shape[-2] - probs.shape[-1]
    if padding > 0:
        decoded1 = decoded1[:, :, padding // 2 : -padding // 2, padding // 2 : -padding // 2]
        decoded2 = decoded2[:, :, padding // 2 : -padding // 2, padding // 2 : -padding // 2]

    if cfgh["masking"]:
        decoded1 = Lambda(lambda x: x * mask, name="amnp")(decoded1)
        decoded2 = Lambda(lambda x: x * mask, name="phi")(decoded2)

    # forward propagation
    obj = CombineComplex()(decoded1, decoded2)

    # masked_obj = Lambda(lambda x: x[0] * tf.cast(x[1], tf.complex64), name="masked_obj")([obj, mask])

    if "single" in cfgh["probe_mode"]:
        probe_lr = tf.constant(probs[None, ...], dtype="complex64")
    else:
        probe_lr = tf.constant(probs[:, None], dtype="complex64")

    # probe function propagation to get the diff
    Psi = Lambda(lambda x: ff_propagation(x[0], x[1]), name="farfield_diff")([probe_lr, obj])

    # Put together
    autoencoder = Model(input_img, [Psi, decoded1, decoded2])
    return autoencoder
