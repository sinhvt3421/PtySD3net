import tensorflow as tf
import numpy as np


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


# def combine_complex(amp, phi):
#     import tensorflow as tf

#     output = tf.cast(amp, tf.complex64) * tf.exp(1j * tf.cast(phi, tf.complex64))
#     return output


def combine_complex(amp, phi):
    output = tf.complex(amp * tf.math.cos(phi), amp * tf.math.sin(phi))
    return output


def ff_propagation(data, const, mode="2d"):
    import tensorflow as tf

    """
    diffraction. Assume same x and y lengthss and uniform sampling
        data:        source plane field
        
    """
    if mode == "2d":
        diff = fourier_transform(data)
    else:
        diff = fourier_transform_3d(data)

    # far-field amplitude
    intensity = tf.math.abs(diff)

    intensity = tf.cast(intensity, tf.float32)

    return intensity / const


def invert_ff_propagation(data, intensity, const):
    import tensorflow as tf

    diff = fourier_transform(data) / const

    intensity_diff = tf.math.abs(diff)

    diff = tf.where(intensity >= 0.0, tf.cast(intensity / intensity_diff, "complex64") * diff, diff)
    # diff = combine_complex(intensity , tf.math.angle(diff))

    exitw = tf.signal.ifft2d(tf.signal.fftshift(diff)) * const

    return intensity_diff, exitw


# 3D fourier transform
def fourier_transform_3d(input):
    import tensorflow as tf

    # fft3d transform with channel unequal to 1
    perm_input = tf.transpose(input, [0, 3, 1, 2])
    perm_fr = tf.signal.fftshift(tf.signal.fft2d(perm_input))
    fr = tf.transpose(perm_fr, [0, 2, 3, 1])
    return fr


# 2D fourier transform
def fourier_transform(input):
    import tensorflow as tf

    # fft3d transform with channel unequal to 1
    fr = tf.signal.fftshift(tf.signal.fft2d(input))
    return fr


def get_kernel_list(size):
    size_half = np.floor(size / 2.0).astype(int)

    r = np.zeros([size])
    r[:size_half] = np.arange(size_half) + 1
    r[size_half:] = np.arange(size_half, 0, -1)

    c = np.zeros([size])
    c[:size_half] = np.arange(size_half) + 1
    c[size_half:] = np.arange(size_half, 0, -1)

    [R, C] = np.meshgrid(r, c)

    help_index = np.round(np.sqrt(R**2 + C**2))
    kernel_list = []

    for i in range(1, size_half + 1):
        new_matrix = np.zeros(shape=[size, size])
        new_matrix[help_index == i] = 1
        kernel_list.append(new_matrix)

    return tf.expand_dims(tf.constant(kernel_list, dtype=tf.complex64), 1)


@tf.function()
@tf.autograph.experimental.do_not_convert
def FRC_loss(i1, i2, kernel_list=get_kernel_list(236)):
    i1 = tf.cast(i1, dtype=tf.complex64)
    i2 = tf.cast(i2, dtype=tf.complex64)

    I1 = tf.signal.fft2d(i1)
    I2 = tf.signal.fft2d(i2)

    A = tf.multiply(I1, tf.math.conj(I2))
    B = tf.multiply(I1, tf.math.conj(I1))
    C = tf.multiply(I2, tf.math.conj(I2))

    A_val = tf.reduce_mean(tf.multiply(tf.expand_dims(A, 0), kernel_list), axis=(-1, -2))
    B_val = tf.reduce_mean(tf.multiply(tf.expand_dims(B, 0), kernel_list), axis=(-1, -2))
    C_val = tf.reduce_mean(tf.multiply(tf.expand_dims(C, 0), kernel_list), axis=(-1, -2))

    den = tf.sqrt(tf.abs(tf.multiply(B_val, C_val)))

    res = tf.abs(A_val) / den

    res = tf.where(tf.compat.v1.is_inf(res), tf.zeros_like(res), res)  # inf
    res = tf.where(tf.compat.v1.is_nan(res), tf.zeros_like(res), res)  # nan

    return 1.0 - tf.reduce_sum(tf.reduce_mean(res, -1)) / 118.0


def SSIMMetric(max_val=1.0):
    def ssim(y_true, y_pred):
        if len(tf.shape(y_true)) > 3:
            return tf.reduce_mean(
                tf.image.ssim(
                    tf.transpose(y_true, [0, 2, 3, 1]),
                    tf.transpose(y_pred, [0, 2, 3, 1]),
                    max_val,
                )
            )
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val))

    return ssim


def PSNRMetric(max_val=1.0):
    def psnr(y_true, y_pred):
        if len(tf.shape(y_true)) > 3:
            return tf.reduce_mean(
                tf.image.psnr(
                    tf.transpose(y_true, [0, 2, 3, 1]),
                    tf.transpose(y_pred, [0, 2, 3, 1]),
                    max_val,
                )
            )
        return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val))

    return psnr


def SSIMLoss(max_val=1.0):
    def ssimloss(y_true, y_pred):
        return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val))

    return ssimloss


def PSNRLoss(max_val=1.0):
    def psnrloss(y_true, y_pred):
        return -tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val))

    return psnrloss


def masked_MSEloss(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    masked_squared_error = tf.square(y_pred - y_true)
    masked_mse = tf.reduce_sum(masked_squared_error) / tf.reduce_sum(mask)
    return masked_mse


def total_var(images):
    ndims = len(tf.shape(images))
    if ndims == 4:  # [B, T, H, W]
        pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
        pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]
        sum_axis = [1, 2, 3]
    total_vars = tf.reduce_sum(tf.abs(pixel_dif1), axis=sum_axis) + tf.reduce_sum(tf.abs(pixel_dif2), axis=sum_axis)

    return total_vars


@tf.function
def loss_tv(Y_true, Y_pred):
    loss_1 = loss_mae(Y_true, Y_pred)
    loss_2 = total_var(Y_pred)
    a1 = 1
    a2 = 0.5
    loss_value = a1 * loss_1 + a2 * loss_2
    return loss_value


@tf.function()
def loss_mae(Y_true, Y_pred):
    top = tf.reduce_sum(tf.math.abs(Y_pred - Y_true), axis=(1, 2, 3), keepdims=True)
    bottom = tf.reduce_sum(tf.math.abs(Y_true), axis=(1, 2, 3), keepdims=True)
    loss_value = tf.reduce_sum(top / bottom)
    return loss_value


@tf.function
def loss_comb(Y_true, Y_pred):
    loss_1 = loss_mae(Y_true, Y_pred)
    loss_2 = tf.keras.losses.MeanSquaredError()(Y_true, Y_pred)
    a1 = 10
    a2 = 1
    loss_value = a1 * loss_1 + a2 * loss_2
    return loss_value
