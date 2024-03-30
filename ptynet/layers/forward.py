import tensorflow as tf
import numpy as np


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def combine_complex(amp, phi):
    output = tf.complex(amp * tf.math.cos(phi), amp * tf.math.sin(phi))
    return output
