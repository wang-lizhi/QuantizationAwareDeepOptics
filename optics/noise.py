import tensorflow as tf


def gaussian_noise(image, std_dev=0.001):
    return image + tf.random.normal(tf.shape(image), 0.0, std_dev, dtype=image.dtype)
