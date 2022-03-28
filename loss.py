import tensorflow as tf


def ssim_loss(ground_truth, prediction):
    return 1 - tf.reduce_mean(tf.image.ssim(ground_truth, prediction, max_val=1))


def log_loss(ground_truth, prediction):
    loss = tf.math.square(tf.math.log(ground_truth + 1) - tf.math.log(prediction + 1))
    return loss


LOSS_FUNCTION_FILTER = {
    "mse": "mse",
    "mae": "mae",
    "ssim": ssim_loss,
    "log": log_loss
}
