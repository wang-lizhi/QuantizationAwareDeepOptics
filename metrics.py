import tensorflow as tf


def ssim_metric(ground_truth, prediction):
    return tf.reduce_mean(tf.image.ssim(ground_truth, prediction, max_val=1.0))


def psnr_metric(ground_truth, prediction):
    mse = tf.reduce_mean(tf.square(ground_truth - prediction))
    psnr = tf.math.subtract(
        (20.0 * tf.math.log(1.0) / tf.math.log(10.0)),
        ((10.0 / tf.math.log(10.0)) * tf.math.log(mse)))
    return psnr


def psnr_tf_metric(ground_truth, prediction):
    return tf.reduce_mean(tf.image.psnr(ground_truth, prediction, max_val=1.0))


def psnr_hyper_metric(ground_truth, prediction):
    ground_truth_channels = tf.split(ground_truth, num_or_size_splits=31, axis=-1)
    prediction_channels = tf.split(prediction, num_or_size_splits=31, axis=-1)
    psnr_sum = 0.0
    for i in range(31):
        channel_mse = tf.reduce_mean(tf.square(ground_truth_channels[i] - prediction_channels[i]))
        channel_psnr = tf.math.subtract(
            (20.0 * tf.math.log(tf.reduce_max(ground_truth_channels[i])) / tf.math.log(10.0)),
            ((10.0 / tf.math.log(10.0)) * tf.math.log(channel_mse)))
        psnr_sum += channel_psnr
    return tf.math.divide(psnr_sum, 31.0)


def sam_metric(ground_truth, prediction):
    numerator = tf.reduce_sum(tf.multiply(prediction, ground_truth), axis=-1)
    denominator = tf.linalg.norm(ground_truth, axis=-1) * tf.linalg.norm(prediction, axis=-1)
    cosine_val = tf.clip_by_value(tf.math.divide_no_nan(numerator, denominator), clip_value_min=-1, clip_value_max=1)
    sam_angles = tf.acos(cosine_val)
    return tf.reduce_mean(sam_angles)


def ergas_metric(ground_truth, prediction):
    ground_truth_channels = tf.split(ground_truth, num_or_size_splits=31, axis=-1)
    prediction_channels = tf.split(prediction, num_or_size_splits=31, axis=-1)
    _ergas = 0.0
    for i in range(31):
        _ergas += tf.reduce_mean(tf.square(prediction_channels[i] - ground_truth_channels[i])) / \
                  tf.square(tf.reduce_mean(ground_truth_channels[i]))
    return 100*tf.sqrt(_ergas / 31)


def psnr_gt_max_metric(ground_truth, prediction):
    return tf.reduce_mean(tf.image.psnr(ground_truth, prediction, max_val=tf.reduce_max(ground_truth)))


def unpack_reconstruction_from_prediction(prediction):
    reconstruction, _, _ = tf.split(prediction, num_or_size_splits=3, axis=-1)
    reconstruction = tf.squeeze(reconstruction, axis=-1)
    return reconstruction


def unpack_element_from_prediction(prediction, pos=0):
    elements = tf.split(prediction, num_or_size_splits=3, axis=-1)
    unpacked = elements[pos]
    unpacked = tf.squeeze(unpacked, axis=-1)
    return unpacked


def post_recheck_ssim_metric(ground_truth, prediction):
    return ssim_metric(ground_truth, unpack_reconstruction_from_prediction(prediction))


def post_recheck_psnr_metric(ground_truth, prediction):
    return psnr_metric(ground_truth, unpack_reconstruction_from_prediction(prediction))


def post_recheck_psnr_hyper_metric(ground_truth, prediction):
    return psnr_hyper_metric(ground_truth, unpack_reconstruction_from_prediction(prediction))


def post_recheck_sam_metric(ground_truth, prediction):
    return sam_metric(ground_truth, unpack_reconstruction_from_prediction(prediction))
