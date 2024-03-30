import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from .base_layers import Conv_Down_block_3D_c, Conv_Down_block


class TBEncoder(keras.layers.Layer):
    def __init__(self, n_layers=4, filters=8, w=3, k_pool=2, pool="max", activation="swish", name="", **kwargs):
        super(TBEncoder, self).__init__(name=name, **kwargs)

        self.tb_down = [
            Conv_Down_block_3D_c(filters * 2**i, w, k_pool, pool=pool, act=activation, name="tb_encoder_{}".format(i))
            for i in range(n_layers)
        ]

        self.latent = Conv_Down_block_3D_c(filters * 2 ** (n_layers - 1), w, act=activation, pool=None, name="latent")

    def call(self, x):
        for i in range(len(self.tb_down)):
            x = self.tb_down[i](x)
        x = self.latent(x)
        return x


class CNNEncoder(keras.layers.Layer):
    def __init__(self, n_layers=4, filters=8, w=3, k_pool=2, pool="max", activation="swish", name="", **kwargs):
        super(CNNEncoder, self).__init__(name=name, **kwargs)

        self.cnn_down = [
            Conv_Down_block(filters * 2**i, w, k_pool, pool=pool, act=activation, name="encoder_{}".format(i))
            for i in range(n_layers)
        ]

        self.latent = Conv_Down_block(filters * 2 ** (n_layers - 1), w, act=activation, pool=None, name="latent")

    def call(self, x):
        for i in range(len(self.cnn_down)):
            x = self.cnn_down[i](x)
        x = self.latent(x)
        return x
