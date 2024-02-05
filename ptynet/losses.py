import tensorflow as tf
import tensorflow.keras.backend as K


def log10(x):
    x1 = tf.math.log(x)
    x2 = tf.math.log(10.0)
    return x1 / x2


def negative_log_loss(y_true, y_pred):
    norm = tf.math.floor(log10(tf.reduce_max(y_true)))
    return -y_pred.log_prob(tf.where(y_true > 5, y_true, 0)) / tf.pow(10.0, norm)
    # return -y_pred.log_prob(y_true) / tf.pow(10.0, norm)


def r_f_I(y_true, y_pred):
    masked_squared_error = tf.square(tf.sqrt(y_pred) - tf.sqrt(y_true))
    masked_mse = tf.reduce_sum(masked_squared_error) / tf.reduce_sum(y_true)
    return masked_mse


def r_f_a(y_true, y_pred):
    masked_squared_error = tf.abs(y_pred - y_true)
    masked_mse = tf.reduce_sum(masked_squared_error) / tf.reduce_sum(y_true)
    return masked_mse


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def total_var(images):
    # ndims = len(tf.shape(images))
    # if ndims == 4:  # [B, T, H, W]
    pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
    pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]
    sum_axis = [2, 3]
    total_vars = tf.reduce_sum(tf.abs(pixel_dif1), axis=sum_axis) + tf.reduce_sum(tf.abs(pixel_dif2), axis=sum_axis)

    # pixel_dif3 = images[:, 1:] - images[:, :-1]
    # sum_axis = [2, 3]
    # total_vars_t = tf.reduce_sum(tf.abs(pixel_dif3), axis=sum_axis)+ tf.reduce_sum(total_vars_t))

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


def masked_MSEloss_v(value=1.0):
    def loss(y_true, y_pred):
        if value != 0:
            mask = tf.cast(tf.logical_and(tf.not_equal(y_true, 0), tf.not_equal(y_true, value)), tf.float32)
            masked_squared_error = tf.square(y_pred - y_true)
            masked_mse2 = tf.reduce_sum(masked_squared_error) / tf.reduce_sum(mask)
            return masked_mse2
        else:
            mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
            masked_squared_error = tf.square(y_pred - y_true)
            masked_mse = tf.reduce_sum(masked_squared_error) / tf.reduce_sum(mask)
            return masked_mse

    return loss


def SEloss(y_true, y_pred):
    masked_squared_error = tf.square(y_pred - y_true)
    masked_mse = tf.sqrt(tf.reduce_sum(masked_squared_error)) / tf.sqrt(tf.reduce_sum(tf.square(y_true)))
    return masked_mse


def SEloss_v(value=1.0):
    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, value), tf.float32)
        masked_squared_error = tf.square(y_pred * mask - y_true * mask)
        masked_mse2 = tf.sqrt(tf.reduce_sum(masked_squared_error)) / tf.sqrt(tf.reduce_sum(tf.square(y_true * mask)))
        return masked_mse2

    return loss


def pearson_correlation_loss(y_true, y_pred):
    # Flatten the images along the spatial dimensions
    y_true_flat = tf.reshape(y_true, (tf.shape(y_true)[0] * tf.shape(y_true)[1], -1))
    y_pred_flat = tf.reshape(y_pred, (tf.shape(y_pred)[0] * tf.shape(y_true)[1], -1))

    # Calculate mean for each image in the batch
    mean_true = tf.reduce_mean(y_true_flat, axis=1, keepdims=True)
    mean_pred = tf.reduce_mean(y_pred_flat, axis=1, keepdims=True)

    # Calculate Pearson correlation coefficient
    numerator = tf.reduce_sum(tf.abs(y_true_flat - mean_true) * tf.abs(y_pred_flat - mean_pred), axis=1)
    denominator_true = tf.sqrt(tf.reduce_sum(tf.square(y_true_flat - mean_true), axis=1))
    denominator_pred = tf.sqrt(tf.reduce_sum(tf.square(y_pred_flat - mean_pred), axis=1))

    correlation_coefficient = numerator / (denominator_true * denominator_pred + 1e-9)

    # The loss is defined as 1 - correlation_coefficient
    loss = 1.0 - tf.reduce_mean(correlation_coefficient)

    return loss
