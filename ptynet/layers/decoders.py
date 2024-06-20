import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Lambda, Conv3D, MultiHeadAttention, LayerNormalization, Dense, Flatten
from .base_layers import Conv_Up_Temporal_Block, Conv_Up_block


class TBDecoder(keras.layers.Layer):
    def __init__(self, n_layers=4, filters=8, w=3, activation="swish", name="", **kwargs):
        super(TBDecoder, self).__init__(name=name, **kwargs)

        self.tb_up = [
            Conv_Up_Temporal_Block(filters / 2 * 2 ** (n_layers - i), w, act=activation, name="tb_decoder_{}".format(i))
            for i in range(n_layers - 1)
        ]

        self.tb_up_last = Conv_Up_Temporal_Block(filters, w, name="decoder_{}".format(n_layers - 1))
        self.out = Conv3D(filters, (1, w, w), padding="same", activation="swish")

    def call(self, x):
        for i in range(len(self.tb_up)):
            x = self.tb_up[i](x)
        x = self.tb_up_last(x)
        x = self.out(x)
        return x


class CNNDecoder(keras.layers.Layer):
    def __init__(self, n_layers=4, filters=8, w=3, activation="swish", name="", **kwargs):
        super(CNNDecoder, self).__init__(name=name, **kwargs)

        self.tb_up = [
            Conv_Up_block(filters / 2 * 2 ** (n_layers - i), w, act=activation, name="decoder_{}".format(i))
            for i in range(n_layers - 1)
        ]

        self.tb_up_last = Conv_Up_block(filters, w, name="decoder_{}".format(n_layers - 1))
        self.out = Conv2D(filters, w, padding="same", activation="swish")

    def call(self, x):
        for i in range(len(self.tb_up)):
            x = self.tb_up[i](x)
        x = self.tb_up_last(x)
        x = self.out(x)
        return x
