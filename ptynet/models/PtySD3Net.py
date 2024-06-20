import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda, Conv3D, Conv2D, Concatenate
from ptynet.layers import *
from tensorflow.keras.callbacks import *
from ptynet.models import PtyBase
from ptynet.losses import total_var_3d, total_var

import numpy as np
import tensorflow_probability as tfp

tfpl = tfp.layers
tfd = tfp.distributions


class PtySD3Net(PtyBase):
    def __init__(self, config, pretrained=""):
        model = create_model(config)
        if pretrained:
            print("Load pretrained model from ", pretrained)
            model.load_weights(pretrained).expect_partial()
        super(PtySD3Net, self).__init__(config=config, model=model)


def create_model(config):
    cfgm = config["model"]
    cfgh = config["hyper"]

    probs = np.load(cfgh["probe"], allow_pickle=True)

    if cfgh["probe_norm"]:
        # Scale probe amplitude base on exposure time, current 1s normalized
        probs = tf.constant(probs * np.sqrt(float(cfgh["probe_norm"])), dtype="complex64")

    if cfgh["masking"]:
        # Masking probe position
        mask = np.load(cfgh["masking"], allow_pickle=True)[None, ...]
        mask = tf.constant(mask, dtype="float32")

    diff = Input(name="diff", shape=(None, cfgm["img_size"], cfgm["img_size"], 1), dtype="float32")

    # Encoder
    e = diff

    latent = TBEncoder(
        n_layers=cfgm["n_cov"],
        filters=cfgm["filters"],
        w=cfgm["kernel"],
        k_pool=cfgm["k_pool"],
        pool=cfgm["pool"],
        name="encoder_tb",
    )(e)

    # Decoder if cfgh["n_refine"] else "relu"
    da = TBDecoder(n_layers=cfgm["n_dcov"], filters=cfgm["filters"], w=cfgm["kernel"], name="decoder_amp")(latent)
    a = Conv3D(1, (1, 1, 1), padding="same", activation="sigmoid")(da)
    a = Lambda(lambda x: tf.squeeze(x, -1), name="amp")(a)

    dp = TBDecoder(n_layers=cfgm["n_dcov"], filters=cfgm["filters"], w=cfgm["kernel"], name="decoder_phase")(latent)
    p = Conv3D(1, (1, 1, 1), padding="same", activation=mpi if cfgh["n_refine"] else None)(dp)
    if not cfgh["n_refine"]:
        p = Mpi()(p)
    p = Lambda(lambda x: tf.squeeze(x, -1), name="phi")(p)

    # Cropping objects, diffraction to match probs shape
    padding = diff.shape[-2] - probs.shape[-1]

    if padding == 0:
        diff_pad = diff[..., 0]
    else:
        diff_pad = diff[:, :, padding // 2 : -padding // 2, padding // 2 : -padding // 2, 0]
        a = a[:, :, padding // 2 : -padding // 2, padding // 2 : -padding // 2]
        p = p[:, :, padding // 2 : -padding // 2, padding // 2 : -padding // 2]

    if cfgh["masking"]:
        a = Lambda(lambda x: x, name="amplitude")(a)
        p = Lambda(lambda x: x * mask, name="phase")(p)

    if cfgh["n_refine"]:
        a = TV(0.1, "tv_a")(a)
        p = TV(0.1, "tv_p")(p)

    objects = CombineComplex()(a, p)

    # Refinement Block
    if "single" in cfgh["probe_mode"]:
        probe_lr = tf.constant(probs[None, ...], dtype="complex64")
    else:
        probe_lr = tf.constant(probs[:, None], dtype="complex64")

    Refine = RefineLayer(mask if cfgh["masking"] else None, cfgh["n_refine"], cfgh["probe_mode"])

    diff_amp_r, objects_r = Refine(objects, diff_pad, probe_lr, probs.shape[-1])

    # Using Poisson output distribution option or Numerical only
    if cfgh["dist"]:
        diff_intensity_poiss = tfpl.DistributionLambda(
            lambda x: tfd.Independent(tfd.Poisson(x**2), 2),
            name="diff_intensity_poiss",
        )(diff_amp_r)

        output = [diff_intensity_poiss]
    else:
        diff_intensity = Lambda(lambda x: x**2, name="diff_intensity")(diff_amp_r)
        output = [diff_intensity]

    # Masking for output
    if cfgh["masking"]:
        ar = Lambda(lambda x: tf.math.abs(x) * mask, name="amplitude_r")(objects_r)
        pr = Lambda(lambda x: tf.math.angle(x) * mask, name="phase_r")(objects_r)
    else:
        ar = Lambda(lambda x: tf.math.abs(x), name="amplitude_r")(objects_r)
        pr = Lambda(lambda x: tf.math.angle(x), name="phase_r")(objects_r)

    ar = TV(0.1, "tv_ar")(ar)
    pr = TV(1.0, "tv_pr", True)(pr)

    output.extend([ar, pr])

    # Training unsupervised and Inference mode options
    model = Model(inputs=diff, outputs=output)

    return model
