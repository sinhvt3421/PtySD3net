import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda, Conv3D, Concatenate
from ptynet.layers import *
from tensorflow.keras.callbacks import *
from ptynet.models import PtyBase
from ptynet.layers.forward import combine_complex
from ptynet.losses import total_var_3d

import numpy as np
import tensorflow_probability as tfp
from tensorflow.signal import fft2d, fftshift

tfpl = tfp.layers
tfd = tfp.distributions


class PtySPINet(PtyBase):
    def __init__(self, config, pretrained="", mode="train"):
        model = create_model(config, mode=mode)
        if pretrained:
            print("Load pretrained model from ", pretrained)
            model.load_weights(pretrained).expect_partial()
        super(PtySPINet, self).__init__(config=config, model=model)


def create_model(config, mode="train"):
    cfgm = config["model"]
    cfgh = config["hyper"]

    probs = np.load(cfgh["probe"], allow_pickle=True)

    if cfgh["probe_norm"]:
        probs = probs * np.sqrt(float(cfgh["probe_norm"]) / np.sum(np.abs(probs) ** 2))

    if cfgh["masking"]:
        mask = np.load(cfgh["masking"], allow_pickle=True)[None, :, :].astype(np.float32)
        mask = tf.constant(mask)

    diff = Input(name="diff", shape=(None, cfgm["img_size"], cfgm["img_size"], 1), dtype="float32")

    e = diff

    skip = []
    for i in range(cfgm["n_cov"]):
        e = Conv_Down_block_3D_c(
            cfgm["filters"] * 2**i, cfgm["kernel"], cfgm["k_pool"], pool=cfgm["pool"], name="encoder_{}".format(i)
        )(e)
        skip.append(e)

    latent = Conv_Down_block_3D_c(
        cfgm["filters"] * 2 ** (cfgm["n_cov"] - 1), cfgm["kernel"], cfgm["k_pool"], pool=False, name="latent"
    )(e)

    da = Conv_Up_block_3D_c(cfgm["filters"] / 2 * 2 ** cfgm["n_dcov"], cfgm["kernel"], name="decoder_a_0")(latent)
    # da = Concatenate()([da, skip[-2]])

    dp = Conv_Up_block_3D_c(cfgm["filters"] / 2 * 2 ** cfgm["n_dcov"], cfgm["kernel"], name="decoder_p_0")(latent)
    # dp = Concatenate()([dp, skip[-2]])

    for i in range(1, cfgm["n_dcov"] - 1):
        da = Conv_Up_block_3D_c(
            cfgm["filters"] / 2 * 2 ** (cfgm["n_dcov"] - i), cfgm["kernel"], name="decoder_a_{}".format(i)
        )(da)
        dp = Conv_Up_block_3D_c(
            cfgm["filters"] / 2 * 2 ** (cfgm["n_dcov"] - i), cfgm["kernel"], name="decoder_p_{}".format(i)
        )(dp)
        # if i < cfgm["n_dcov"] - 2:
        #     da = Concatenate()([da, skip[-2 - i]])
        #     dp = Concatenate()([dp, skip[-2 - i]])

    da = Conv_Up_block_3D_c(cfgm["filters"], cfgm["kernel"], name="decoder_a_{}".format(cfgm["n_dcov"] - 1))(da)
    da = Conv3D(cfgm["filters"], (1, cfgm["kernel"], cfgm["kernel"]), padding="same", activation="swish")(da)
    a = Conv3D(1, (1, 1, 1), padding="same", activation="sigmoid")(da)
    a = Lambda(lambda x: tf.squeeze(x, -1), name="amp")(a)

    dp = Conv_Up_block_3D_c(cfgm["filters"], cfgm["kernel"], name="decoder_p_{}".format(cfgm["n_dcov"] - 1))(dp)
    dp = Conv3D(cfgm["filters"], (1, cfgm["kernel"], cfgm["kernel"]), padding="same", activation="swish")(dp)
    p = Conv3D(1, (1, 1, 1), padding="same", activation=mpi)(dp)
    p = Lambda(lambda x: tf.squeeze(x, -1), name="phi")(p)

    if cfgh["masking"]:
        a = Lambda(lambda x: x, name="amplitude")(a)
        p = Lambda(lambda x: x * mask, name="phase")(p)

    if cfgh["probe_mode"] == "single":
        probe_lr = ProbeLayer(probs[None, ...], False)
    else:
        probe_lr = ProbeLayer(probs[:, None], False)

    Refine = RefineLayer(probe_lr, mask if cfgh["masking"] else None, cfgh["n_refine"])

    objects = combine_complex(a, p)

    padding = diff.shape[-2] - probs.shape[-1]

    if padding < 0:
        diff_pad = tf.pad(
            diff[..., 0], ((0, 0), (0, 0), (-padding // 2, -padding // 2), (-padding // 2, -padding // 2))
        )
    elif padding == 0:
        diff_pad = diff[..., 0]
    else:
        diff_pad = diff[:, :, padding // 2 : -padding // 2, padding // 2 : -padding // 2, 0]
        objects = objects[:, :, padding // 2 : -padding // 2, padding // 2 : -padding // 2]

    print(padding, probs.shape, diff_pad.shape, objects.shape, np.max(np.abs(probs), axis=(-1, -2)))

    if cfgh["probe_mode"] == "single":
        pre_exit = probe_lr(objects)

        dif = fftshift(fft2d(pre_exit / probs.shape[-1]), axes=(-2, -1))
        diff_amp = tf.abs(dif)
    else:
        pre_exit = probe_lr(tf.expand_dims(objects, 1))

        dif = fftshift(fft2d(pre_exit / probs.shape[-1]), axes=(-2, -1))

        diff_amp = tf.sqrt(tf.reduce_sum(tf.abs(dif) ** 2, 1))

    diff_amp_r, objects_r, new_probe = Refine(objects, diff_pad, probs.shape[-1], cfgh["probe_mode"])

    if cfgh["masking"]:
        ar = Lambda(lambda x: tf.math.abs(x), name="amplitude_r")(objects_r)
        pr = Lambda(lambda x: tf.math.angle(x) * mask, name="phase_r")(objects_r)
    else:
        ar = Lambda(lambda x: tf.math.abs(x), name="amplitude_r")(objects_r)
        pr = Lambda(lambda x: tf.math.angle(x), name="phase_r")(objects_r)

    diff_intensity = Lambda(lambda x: x**2, name="diff_intensity")(diff_amp)

    if cfgh["dist"]:
        diff_intensity_r = tfpl.DistributionLambda(
            lambda x: tfd.Independent(tfd.Poisson(x**2), 2),
            name="diff_intensity_r",
        )(diff_amp_r)

        output = diff_intensity_r
    else:
        output = diff_intensity

    if mode == "train":
        model = Model(inputs=diff, outputs=output)
        model.add_loss(0.001 * total_var_3d(pr))
    else:
        model = Model(inputs=diff, outputs=[ar, pr])

    return model
