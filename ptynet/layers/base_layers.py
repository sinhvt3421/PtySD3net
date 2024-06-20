import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Conv2DTranspose,
    Conv3D,
    MaxPool3D,
    Conv3DTranspose,
    UpSampling3D,
    Activation,
)
import math
from tensorflow.keras.layers import BatchNormalization, LayerNormalization

import tensorflow.keras as keras
from tensorflow.keras.regularizers import l2
from tensorflow.signal import fft2d, fftshift, ifftshift, ifft2d
from ptynet.losses import total_var_3d


def mpi(input_tensor):
    """
    A modified version of tanh with limit range to [-pi,pi]
    """

    return tf.tanh(input_tensor) * (math.pi)

class Mpi(tf.keras.layers.Layer):
    """
        A modified version of tanh with trainable parameters limit range to [-pi,pi]
    """
    def __init__(self, **kwargs):
        super(Mpi, self).__init__(**kwargs)
        self.alpha = tf.Variable(
            0.5, name="alpha_act", trainable=True, constraint=lambda x: tf.clip_by_value(x, -math.pi, math.pi)
        )

    def call(self, inputs):
        return tf.math.tanh(inputs) * self.alpha

def combine_complex(amp, phi):
    return tf.cast(amp, tf.complex64) * tf.exp(1j * tf.cast(phi, tf.complex64))


class CombineComplex(tf.keras.layers.Layer):
    def call(self, amp, phi):
        return combine_complex(amp,phi)

class TV(tf.keras.layers.Layer):
    def __init__(self, gama, name="TV", train=False, **kwargs):
        super(TV, self).__init__(name=name, **kwargs)
        self.gama = tf.Variable(
            gama,
            name="gama_{}".format(name),
            trainable=train,
            constraint=lambda x: tf.clip_by_value(x, 0.1, 2.0),
        )

    def call(self, inputs):
        self.add_loss(self.gama * total_var_3d(inputs))
        return inputs


# Define a custom TB layer for CNN update
class CNNTBLayer(tf.keras.layers.Layer):
    def __init__(self, nfilters=32, w=3, dept=1, act="swish", out="sigmoid", name="", **kwargs):
        super(CNNTBLayer, self).__init__(name=name, **kwargs)
        self.cv = [Conv_Up_Temporal_Block(nfilters, w, padding="same", act=act, pool=False) for i in range(dept)]

        self.cv_out = Conv3D(1, (1, w, w), padding="same", activation=None)

        self.act_out = Activation("sigmoid") if out == "sigmoid" else Mpi()

    def call(self, inputs):
        if len(tf.shape(inputs)) == 4:
            x = tf.expand_dims(inputs, -1)
        else:
            x = inputs
        for i in range(len(self.cv)):
            x = self.cv[i](x)

        return tf.squeeze(self.act_out(self.cv_out(x)), -1)


# Define a custom layer for probe function
class RefineLayer(tf.keras.layers.Layer):
    def __init__(self, mask, n_step=5, probe_mode="multi_c", **kwargs):
        super(RefineLayer, self).__init__(**kwargs)
        self.mask = mask

        if n_step > 0:
            self.alpha = tf.Variable(0.1, trainable=True, dtype="float32", name="alpha")

        # self.probe_scale = tf.Variable(1.0, trainable=True, dtype="float32", name="probe_scale")

        self.n_step = n_step
        self.probe_mode = probe_mode

        if "c" in self.probe_mode:
            self.cnn_tb_a = CNNTBLayer(out="sigmoid")
            self.cnn_tb_p = CNNTBLayer(out="")

    def call(self, objects, org_intensity, probe, fftconst):
        """_summary_

        Args:
            objects (_type_): shape [B,T,H,W]
            org_intensity (_type_): shape [B,T,H,W]
        """
        # probe = probe * tf.cast(self.probe_scale, "complex64")

        if "single" in self.probe_mode:
            prob_tf_abs = tf.cast(tf.reduce_max(tf.abs(probe) ** 2.0), "complex64")
        else:
            prob_tf_abs = tf.cast(tf.reduce_sum(tf.reduce_max(tf.abs(probe) ** 2, axis=(-2, -1)), 0), "complex64")

        for i in range(self.n_step):
            if "single" in self.probe_mode:
                # probe shape [1,H,W], objects shape [B,T,H,W] -> pre_exit shape [B,T,H,W]
                pre_exit = probe * objects

                dif = fftshift(fft2d(pre_exit), axes=(-2, -1)) / fftconst
                intensity = tf.abs(dif)

                dif = tf.where(org_intensity >= 0, tf.cast(org_intensity / intensity, "complex64") * dif, dif)
            else:
                # probe shape [M,1,H,W], objects shape [B,1,T,H,W] -> pre_exit shape [B,M,T,H,W]
                pre_exit = probe * tf.expand_dims(objects, 1)

                dif = fftshift(fft2d(pre_exit), axes=(-2, -1)) / fftconst

                # summing over mode probe, dif shape [B,M,T,H,W]
                intensity = tf.sqrt(tf.reduce_sum(tf.abs(dif) ** 2, 1))
                # intensity shape [B,T,H,W], org_intensity shape [B,T,H,W]
                dif = tf.expand_dims(tf.cast(org_intensity / intensity, "complex64"), 1) * dif

            exitw = ifft2d(ifftshift(dif, axes=(-2, -1))) * fftconst

            # real space costraint
            dexit = exitw - pre_exit

            if self.probe_mode == "single":
                update = tf.cast(self.alpha, "complex64") * tf.math.conj(probe) * dexit / prob_tf_abs
                objects += update

            elif self.probe_mode == "single_c":
                invert_update = tf.math.conj(probe) * dexit / prob_tf_abs

                update_a = self.cnn_tb_a(
                    tf.math.abs(invert_update) * self.mask if self.mask is not None else tf.math.abs(invert_update)
                )

                update_p = self.cnn_tb_p(
                    tf.math.angle(invert_update) * self.mask if self.mask is not None else tf.math.angle(invert_update)
                )

                update = combine_complex(update_a, update_p)

                objects = tf.cast(self.alpha, "complex64") * update + objects

            elif self.probe_mode == "multi":
                invert_update = tf.reduce_sum(tf.math.conj(probe) * dexit, 1) / prob_tf_abs

                update_a = tf.math.abs(invert_update)

                update_p = tf.math.angle(invert_update)
                if self.mask is not None:
                    update_p *= self.mask
                    update_a *= self.mask

                update = combine_complex(update_a, update_p)

                objects = tf.cast(self.alpha, "complex64") * update + objects

            else:
                # dexit shape [B,M,T,H,W], probe shape [M,1,H,W] -> [B,M,T,H,W] -> summing over mode probe, invert_update shape [B,T,H,W]
                invert_update = tf.reduce_sum(tf.math.conj(probe) * dexit, 1) / prob_tf_abs

                update_a = self.cnn_tb_a(tf.math.abs(invert_update) * self.mask if self.mask is not None else tf.math.abs(invert_update))

                update_p = self.cnn_tb_p(
                    tf.math.angle(invert_update) * self.mask if self.mask is not None else tf.math.angle(invert_update)
                )

                update = combine_complex(update_a, update_p)

                objects = tf.cast(self.alpha, "complex64") * update + objects

        if "single" in self.probe_mode:
            pre_exit = probe * objects

            dif = fftshift(fft2d(pre_exit), axes=(-2, -1)) / fftconst
            intensity = tf.abs(dif)
        else:
            pre_exit = probe * tf.expand_dims(objects, 1)

            dif = fftshift(fft2d(pre_exit), axes=(-2, -1)) / fftconst

            intensity = tf.sqrt(tf.reduce_sum(tf.abs(dif) ** 2, 1))

        return intensity, objects


# kernel_regularizer=l2(1e-5)
class Conv_Down_Temporal_Block(keras.layers.Layer):
    def __init__(self, nfilters, w=3, p=2, padding="same", pool=None, act="swish", name="", **kwargs):
        super(Conv_Down_Temporal_Block, self).__init__(name=name, **kwargs)

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
            self.pool = Conv3D(nfilters, (1, p, p), strides=(1, p, p), padding="valid")
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
class Conv_Up_Temporal_Block(keras.layers.Layer):
    def __init__(self, nfilters, w=3, padding="same", trans=True, act="swish", name="", **kwargs):
        super(Conv_Down_Temporal_Block, self).__init__(name=name, **kwargs)

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


# encoder layers
class Conv_Down_block(keras.layers.Layer):
    def __init__(self, nfilters, w=3, p=2, padding="same", pool=None, act="swish", **kwargs):
        super(Conv_Down_block, self).__init__(**kwargs)

        self.cv1 = Conv2D(nfilters, w, padding=padding, activation=act, kernel_regularizer=l2(1e-5))
        self.cv2 = Conv2D(nfilters, w, padding=padding, activation=act, kernel_regularizer=l2(1e-5))

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

