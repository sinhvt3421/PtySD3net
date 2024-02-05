import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Conv2DTranspose,
    Embedding,
    Conv3D,
    MaxPool3D,
    Conv3DTranspose,
    UpSampling2D,
    UpSampling3D,
    Activation,
)
import math
from tensorflow.keras.layers import BatchNormalization, LayerNormalization

import tensorflow.keras as keras
from tensorflow.keras.regularizers import l2
from tensorflow.signal import fft2d, fftshift, ifftshift, ifft2d
from .forward import combine_complex


def mpi(input_tensor):
    """
    A modified version of relu with linear gradient.
    """

    return tf.tanh(input_tensor) * (math.pi)


def lrelu(x, alpha=0.01):
    return tf.nn.leaky_relu(x, alpha)


def total_var(images):
    # ndims = len(tf.shape(images))
    # if ndims == 4:  # [B, T, H, W]
    pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
    pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]
    sum_axis = [2, 3]
    total_vars = tf.reduce_sum(tf.abs(pixel_dif1), axis=sum_axis) + tf.reduce_sum(tf.abs(pixel_dif2), axis=sum_axis)

    return tf.reduce_sum(total_vars) / (2 * tf.cast(tf.shape(images)[-1] ** 2, "float32"))


def total_var_3d(images):
    # ndims = len(tf.shape(images))
    # if ndims == 4:  # [B, T, H, W]
    pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
    pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]
    pixel_dif3 = images[:, 1:, :, :] - images[:, :-1, :, :]

    sum_axis = [2, 3]

    total_vars = tf.reduce_sum(tf.abs(pixel_dif1), axis=sum_axis) + tf.reduce_sum(tf.abs(pixel_dif2), axis=sum_axis)
    total_vars_2 = tf.reduce_sum(tf.abs(pixel_dif3), axis=sum_axis)

    scale = tf.cast(tf.shape(images)[-1] ** 2, "float32")
    time = tf.cast(tf.shape(images)[1], "float32")

    return tf.reduce_sum(total_vars) / (time * 2 * scale) + tf.reduce_sum(total_vars_2) / ((time - 1) * scale)


class Mpi(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mpi, self).__init__(**kwargs)
        self.alpha = tf.Variable(
            0.2, name="alpha_act", trainable=True, constraint=lambda x: tf.clip_by_value(x, -math.pi, math.pi)
        )

    def call(self, inputs):
        return tf.math.tanh(inputs) * self.alpha


# Define a custom layer for CNN update
class CNNLayer(tf.keras.layers.Layer):
    def __init__(self, nfilters=8, w=3, dept=5, act="swish", out="sigmoid", name="", **kwargs):
        super(CNNLayer, self).__init__(name=name, **kwargs)
        self.cv = [Conv3D(nfilters, (1, w, w), padding="same", activation=act) for i in range(dept)]

        self.cv_out = Conv3D(1, (1, w, w), padding="same", activation=None)
        self.norm = BatchNormalization()

        self.act_out = Activation("sigmoid") if out == "sigmoid" else Mpi()

    def call(self, inputs):
        x = tf.expand_dims(inputs, -1)

        for i in range(len(self.cv)):
            x = self.cv[i](x)

        x = self.norm(x)

        return tf.squeeze(self.act_out(self.cv_out(x)), -1)


# Define a custom layer for CNN update
class CNNTBLayer(tf.keras.layers.Layer):
    def __init__(self, nfilters=8, w=3, dept=1, act="swish", out="sigmoid", name="", **kwargs):
        super(CNNTBLayer, self).__init__(name=name, **kwargs)
        self.cv = [Conv_Down_block_3D_c(nfilters, w, padding="same", act=act, pool=False) for i in range(dept)]

        self.cv_out = Conv3D(1, (1, w, w), padding="same", activation=None)

        self.act_out = Activation("sigmoid") if out == "sigmoid" else Mpi()

    def call(self, inputs):
        x = tf.expand_dims(inputs, -1)

        for i in range(len(self.cv)):
            x = self.cv[i](x)

        return tf.squeeze(self.act_out(self.cv_out(x)), -1)


class ProbeLayer(tf.keras.layers.Layer):
    def __init__(self, probe, train=True, **kwargs):
        super(ProbeLayer, self).__init__(**kwargs)
        self.train = train
        if self.train:
            self.probe_abs = tf.Variable(tf.cast(tf.abs(probe), "float32"), trainable=True, name="abs_update")
            self.probe_phase = tf.Variable(
                tf.cast(tf.math.angle(probe), "float32"), trainable=False, name="angle_update"
            )
        else:
            self.probe = tf.Variable(probe, trainable=False, dtype="complex64")
            # self.probe = tf.constant(probe, dtype="complex64")

    def call(self, inputs):
        if self.train:
            return inputs * combine_complex(self.probe_abs, self.probe_phase)
        return inputs * self.probe

    def get_probe(self):
        if self.train:
            return combine_complex(self.probe_abs, self.probe_phase)
        else:
            return self.probe

    def update_probe(self, update):
        self.probe += update


# Define a custom layer for probe function
class RefineLayer(tf.keras.layers.Layer):
    def __init__(self, probe_lr, mask, n_step=5, **kwargs):
        super(RefineLayer, self).__init__(**kwargs)
        self.probe_lr = probe_lr
        self.mask = mask

        self.scale_factor = tf.Variable(0.1, trainable=True, dtype="float32", name="alpha")

        self.n_step = n_step
        # self.cnn_a = CNNLayer(out="sigmoid")
        # self.cnn_p = CNNLayer(out="")

        self.cnn_a = [CNNTBLayer(out="sigmoid") for i in range(self.n_step)]
        self.cnn_p = [CNNTBLayer(out="") for i in range(self.n_step)]

    def call(self, objects, org_intensity, fftconst, mode="multi_c"):
        """_summary_

        Args:
            objects (_type_): shape [B,T,H,W]
            org_intensity (_type_): shape [B,T,H,W]
        """
        if mode == "single":
            prob_tf_abs = tf.cast(tf.reduce_max(tf.abs(self.probe_lr.get_probe()) ** 2.0), "complex64")
        else:
            prob_tf_abs = tf.cast(
                tf.reduce_sum(tf.reduce_max(tf.abs(self.probe_lr.get_probe()) ** 2, axis=(-2, -1)), 0), "complex64"
            )

        for i in range(self.n_step):
            if mode == "single":
                pre_exit = self.probe_lr(objects)

                dif = fftshift(fft2d(pre_exit / fftconst), axes=(-2, -1))
                intensity = tf.abs(dif)

                dif = tf.cast(org_intensity / intensity, "complex64") * dif
            else:
                # probe shape [M,1,H,W], objects shape [B,1,T,H,W] -> pre_exit shape [B,M,T,H,W]
                pre_exit = self.probe_lr(tf.expand_dims(objects, 1))

                dif = fftshift(fft2d(pre_exit / fftconst), axes=(-2, -1))

                # summing over mode probe, dif shape [B,M,T,H,W]
                intensity = tf.sqrt(tf.reduce_sum(tf.abs(dif) ** 2, 1))

                # intensity shape [B,T,H,W], org_intensity shape [B,T,H,W]
                dif = tf.expand_dims(tf.cast(org_intensity / intensity, "complex64"), 1) * dif

            exitw = ifft2d(ifftshift(dif * fftconst, axes=(-2, -1)))

            # real space costraint
            dexit = exitw - pre_exit
            if mode == "single":
                update = (
                    tf.cast(self.scale_factor, "complex64")
                    * tf.math.conj(self.probe_lr.get_probe())
                    * dexit
                    / prob_tf_abs
                )
                objects += update
            elif mode == "multi":
                update = (
                    tf.cast(self.scale_factor, "complex64")
                    * tf.reduce_sum(tf.math.conj(self.probe_lr.get_probe()) * dexit, 1)
                    / prob_tf_abs
                )
                objects += update

            else:
                # dexit shape [B,M,T,H,W], probe shape [M,1,H,W] -> [B,M,T,H,W] -> summing over mode probe, invert_object shape [B,T,H,W]
                invert_object = tf.reduce_sum(tf.math.conj(self.probe_lr.get_probe()) * dexit, 1) / prob_tf_abs

                update_a = self.cnn_a[i](tf.math.abs(invert_object))
                update_p = self.cnn_p[i](tf.math.angle(invert_object))
                if self.mask is not None:
                    update_p *= self.mask

                update = combine_complex(update_a, update_p)

                # # mean over B,T -> prob_up shape [M,H,W]
                # prob_up = (
                #     0.1
                #     * tf.reduce_sum(tf.reduce_mean(tf.math.conj(objects) * dexit, axis=2, keepdims=True), 0)
                #     / tf.cast(tf.reduce_max(tf.abs(objects) ** 2.0), "complex64")
                # )

                # self.probe_lr.update_probe(prob_up)

                objects = tf.cast(self.scale_factor, "complex64") * update + objects

        if mode == "single":
            pre_exit = self.probe_lr(objects)

            dif = fftshift(fft2d(pre_exit / fftconst), axes=(-2, -1))
            intensity = tf.abs(dif)
        else:
            pre_exit = self.probe_lr(tf.expand_dims(objects, 1))

            dif = fftshift(fft2d(pre_exit / fftconst), axes=(-2, -1))

            intensity = tf.sqrt(tf.reduce_sum(tf.abs(dif) ** 2, 1))

        return intensity, objects, self.probe_lr.get_probe()


# encoder layers
class Conv_Down_block(keras.layers.Layer):
    def __init__(self, nfilters, w=3, p=2, padding="same", pool=None, act="swish", **kwargs):
        super(Conv_Down_block, self).__init__(**kwargs)

        self.cv1 = Conv2D(nfilters, w, padding=padding, activation=act)
        self.cv2 = Conv2D(nfilters, w, padding=padding, activation=act)

        if pool == "max":
            self.pool = MaxPool2D(
                p,
                padding=padding,
            )
        elif pool == "stride":
            self.pool = Conv2D(nfilters, w, 2, padding="same", activation=act)
        else:
            self.pool = None

    def call(self, x):
        x = self.cv1(x)
        x = self.cv2(x)

        if self.pool is not None:
            x = self.pool(x)
        return x


# decoder layer
class Conv_Up_block(keras.layers.Layer):
    def __init__(self, nfilters, w=3, padding="same", act="swish", trans=True, **kwargs):
        super(Conv_Up_block, self).__init__(**kwargs)

        self.cv1 = Conv2D(nfilters, w, padding=padding, activation=act, kernel_regularizer=l2(1e-5))
        self.cv2 = Conv2D(nfilters, w, padding=padding, activation=act, kernel_regularizer=l2(1e-5))

        self.tcv = Conv2DTranspose(nfilters, w, strides=2, padding=padding, kernel_regularizer=l2(1e-5))

    def call(self, x):
        x = self.cv1(x)
        x = self.cv2(x)

        x = self.tcv(x)
        return x


class Conv_Down_block_3D_c(keras.layers.Layer):
    def __init__(self, nfilters, w=3, p=2, padding="same", pool=None, act="swish", name="", **kwargs):
        super(Conv_Down_block_3D_c, self).__init__(name=name, **kwargs)

        self.cv = Conv3D(nfilters, (1, 1, 1), padding=padding, activation=act, kernel_regularizer=l2(1e-5))

        self.cv_t1 = Conv3D(nfilters, (1, w, w), padding=padding, activation=act, kernel_regularizer=l2(1e-5))

        self.cv_t2 = Conv3D(nfilters // 2, (3, w, w), padding=padding, activation=act, kernel_regularizer=l2(1e-5))

        self.cv_t3 = Conv3D(nfilters // 2, (5, w, w), padding=padding, activation=act, kernel_regularizer=l2(1e-5))

        self.cv_combine = Conv3D(nfilters, (1, 1, 1), padding=padding, activation=act, kernel_regularizer=l2(1e-5))

        self.norm = BatchNormalization()

        if pool == "max":
            self.pool = MaxPool3D(
                (1, p, p),
                padding=padding,
            )
        elif pool == "stride":
            self.pool = Conv3D(nfilters // 2, (1, p, p), strides=(1, p, p), padding="valid")
        else:
            self.pool = None

    def call(self, x):
        x = self.cv(x)

        x1 = self.cv_t1(x)
        x2 = self.cv_t2(x)
        x3 = self.cv_t3(x)

        x4 = tf.concat([x1, x2, x3], -1)

        x4 = self.cv_combine(x4)
        x4 = self.norm(x4)

        if self.pool is not None:
            x4 = self.pool(x4)

        return x4


# decoder layer
class Conv_Up_block_3D_c(keras.layers.Layer):
    def __init__(self, nfilters, w=3, padding="same", trans=True, act="swish", name="", **kwargs):
        super(Conv_Up_block_3D_c, self).__init__(name=name, **kwargs)

        self.cv = Conv3D(nfilters, (1, 1, 1), padding=padding, activation=act, kernel_regularizer=l2(1e-5))

        self.cv_t1 = Conv3D(nfilters, (1, w, w), padding=padding, activation=act, kernel_regularizer=l2(1e-5))

        self.cv_t2 = Conv3D(nfilters // 2, (3, w, w), padding=padding, activation=act, kernel_regularizer=l2(1e-5))

        self.cv_t3 = Conv3D(nfilters // 2, (5, w, w), padding=padding, activation=act, kernel_regularizer=l2(1e-5))

        self.cv_combine = Conv3D(nfilters, (1, 1, 1), padding=padding, activation=act, kernel_regularizer=l2(1e-5))

        self.norm = BatchNormalization()

        if trans:
            self.tcv = Conv3DTranspose(nfilters, (1, w, w), strides=(1, 2, 2), padding=padding)
        else:
            self.tcv = UpSampling3D(size=(1, 2, 2))

    def call(self, x):
        x = self.cv(x)

        x1 = self.cv_t1(x)
        x2 = self.cv_t2(x)
        x3 = self.cv_t3(x)

        x4 = tf.concat([x1, x2, x3], -1)

        x4 = self.cv_combine(x4)
        x4 = self.norm(x4)

        x4 = self.tcv(x4)

        return x4
