import math

import numpy as np
import tensorflow as tf


def transpose_2d_fft(a_tensor, dtype=tf.complex64):
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_tensor_transp = tf.transpose(a=a_tensor, perm=[0, 3, 1, 2])
    a_fft2d = tf.signal.fft2d(a_tensor_transp)
    a_fft2d = tf.cast(a_fft2d, dtype)
    a_fft2d = tf.transpose(a=a_fft2d, perm=[0, 2, 3, 1])
    return a_fft2d


def transpose_2d_ifft(a_tensor, dtype=tf.complex64):
    a_tensor = tf.transpose(a=a_tensor, perm=[0, 3, 1, 2])
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_ifft2d_transp = tf.signal.ifft2d(a_tensor)
    a_ifft2d = tf.transpose(a=a_ifft2d_transp, perm=[0, 2, 3, 1])
    a_ifft2d = tf.cast(a_ifft2d, dtype)
    return a_ifft2d


def complex_exponent_tf(phase, dtype=tf.complex64, name='complex_exp'):
    # cos(phi) + sin(phi)j
    phase = tf.cast(phase, tf.float64)
    return tf.add(tf.cast(tf.cos(phase), dtype=dtype),
                  1.j * tf.cast(tf.sin(phase), dtype=dtype),
                  name=name)


def laplacian_filter_tf(img_batch):
    laplacian_filter = tf.constant([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=tf.float32)
    laplacian_filter = tf.reshape(laplacian_filter, [3, 3, 1, 1])
    laplacian_filter = tf.cast(laplacian_filter, tf.float32)
    filter_input = tf.cast(img_batch, tf.float32)
    filtered_batch = tf.nn.convolution(input=filter_input, filters=laplacian_filter, padding="SAME")
    return filtered_batch


def laplace_l1_regularizer(scale=100):
    def laplace_l1(a_tensor):
        laplace_filtered = laplacian_filter_tf(a_tensor)
        laplace_filtered = laplace_filtered[:, 1:-1, 1:-1, :]
        return tf.reshape(
            tf.multiply(scale, tf.reduce_mean(input_tensor=tf.abs(laplace_filtered)),
                        name="laplace_l1_regularized"), [])
    return laplace_l1


def laplace_l2_regularizer(scale):
    def laplace_l2(a_tensor):
        with tf.compat.v1.name_scope('laplace_l2_regularizer'):
            laplace_filtered = laplacian_filter_tf(a_tensor)
            laplace_filtered = laplace_filtered[:, 1:-1, 1:-1, :]
            return scale * tf.reduce_mean(input_tensor=tf.square(laplace_filtered))
    return laplace_l2


def get_one_phase_shift_thickness(wave_lengths, refractive_index):
    """
    Calculate the thickness (in meter) of a phase-shift of 2pi.
    """
    delta_n = refractive_index - 1.
    wave_nos = 2. * np.pi / wave_lengths
    two_pi_thickness = (2. * np.pi) / (wave_nos * delta_n)
    return two_pi_thickness


def fft_shift_2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()
    new_tensor = a_tensor
    for axis in range(1, 3):
        split = (input_shape[axis] + 1) // 2
        temp_list = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
        new_tensor = tf.gather(new_tensor, temp_list, axis=axis)
    return new_tensor


def ifft_shift_2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()
    new_tensor = a_tensor
    for axis in range(1, 3):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        temp_list = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, temp_list, axis=axis)
    return new_tensor


def psf2otf(input_filter, output_size):
    """
    Convert 4D tensorflow filter into its FFT.

    :param input_filter: PSF. Shape (height, width, b, c)
    :param output_size: Size of the output OTF.
    :return: The otf.
    """
    fh, fw, _, _ = input_filter.shape.as_list()

    if output_size[0] != fh:
        pad = (output_size[0] - fh) / 2

        if (output_size[0] - fh) % 2 != 0:
            pad_top = pad_left = int(np.ceil(pad))
            pad_bottom = pad_right = int(np.floor(pad))
        else:
            pad_top = pad_left = int(pad) + 1
            pad_bottom = pad_right = int(pad) - 1

        padded = tf.pad(tensor=input_filter, paddings=[[pad_top, pad_bottom],
                                                       [pad_left, pad_right], [0, 0], [0, 0]], mode="CONSTANT")
    else:
        padded = input_filter

    padded = tf.transpose(a=padded, perm=[2, 0, 1, 3])
    padded = ifft_shift_2d_tf(padded)
    padded = tf.transpose(a=padded, perm=[1, 2, 0, 3])

    # FFT
    tmp = tf.transpose(a=padded, perm=[2, 3, 0, 1])
    tmp = tf.signal.fft2d(tf.complex(tmp, 0.))
    return tf.transpose(a=tmp, perm=[2, 3, 0, 1])


def next_power_of_two(number):
    closest_pow = np.power(2, np.ceil(np.math.log(number, 2)))
    return closest_pow


def image_convolve_with_psf(img, psf, otf=None, adjoint=False, img_shape=None):
    img = tf.convert_to_tensor(value=img, dtype=tf.float32)
    psf = tf.convert_to_tensor(value=psf, dtype=tf.float32)
    if img_shape is None:
        img_shape = img.shape.as_list()
    psf_shape = psf.shape.as_list()

    circular = False

    if psf_shape[0] != img_shape[1] or psf_shape[1] != img_shape[2]:
        target_side_length = psf_shape[0]
        height_pad = (target_side_length - img_shape[1]) / 2
        width_pad = (target_side_length - img_shape[1]) / 2

        pad_top, pad_bottom = int(np.ceil(height_pad)), int(np.floor(height_pad))
        pad_left, pad_right = int(np.ceil(width_pad)), int(np.floor(width_pad))
        img = tf.pad(tensor=img, paddings=[[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                     mode="SYMMETRIC")
        img_shape = img.shape.as_list()
    else:
        circular = True

    img_fft = transpose_2d_fft(img)

    if otf is None:
        otf = psf2otf(psf, output_size=img_shape[1:3])
        otf = tf.transpose(a=otf, perm=[2, 0, 1, 3])

    otf = tf.cast(otf, tf.complex64)

    img_fft = tf.cast(img_fft, tf.complex64)

    if adjoint:
        result = transpose_2d_ifft(img_fft * tf.math.conj(otf))
    else:
        result = transpose_2d_ifft(img_fft * otf)
    result = tf.cast(tf.math.real(result), tf.float32)

    if not circular:
        result = result[:, pad_top:-pad_bottom, pad_left:-pad_right, :]
    return result


def depth_dep_convolution(img, psfs, disc_depth_map):
    img = tf.cast(img, dtype=tf.float32)
    input_shape = img.shape.as_list()

    zeros_tensor = tf.zeros_like(img, dtype=tf.float32)
    disc_depth_map = tf.tile(tf.cast(disc_depth_map, tf.int16),
                             multiples=[1, 1, 1, input_shape[3]])
    blurred_imgs = []
    for depth_idx, psf in enumerate(psfs):
        psf = tf.cast(psf, dtype=tf.float32)
        condition = tf.equal(disc_depth_map, tf.convert_to_tensor(value=depth_idx, dtype=tf.int16))
        blurred_img = image_convolve_with_psf(img, psf)
        blurred_imgs.append(tf.where(condition, blurred_img, zeros_tensor))

    result = tf.reduce_sum(input_tensor=blurred_imgs, axis=0)
    return result


def get_spherical_wavefront_phase(resolution, physical_size, wave_lengths, source_distance):
    source_distance = tf.cast(source_distance, tf.float64)
    physical_size = tf.cast(physical_size, tf.float64)
    wave_lengths = tf.cast(wave_lengths, tf.float64)
    N, M = resolution
    [x, y] = np.mgrid[-N // 2:N // 2, -M // 2:M // 2].astype(np.float64)
    x = x / N * physical_size
    y = y / M * physical_size
    curvature = tf.sqrt(x ** 2 + y ** 2 + source_distance ** 2)
    wave_nos = 2. * np.pi / wave_lengths
    phase_shifts = complex_exponent_tf(wave_nos * curvature)
    phase_shifts = tf.expand_dims(tf.expand_dims(phase_shifts, 0), -1)
    return phase_shifts


def least_common_multiple(a, b):
    return abs(a * b) / math.gcd(a, b) if a and b else 0


def area_downsampling_tf(input_image, target_side_length):
    input_shape = input_image.shape.as_list()
    input_image = tf.cast(input_image, tf.float32)

    if not input_shape[1] % target_side_length:
        factor = int(input_shape[1] / target_side_length)
        output_img = tf.nn.avg_pool2d(input=input_image,
                                      ksize=[1, factor, factor, 1],
                                      strides=[1, factor, factor, 1],
                                      padding="VALID")
    else:
        lcm_factor = least_common_multiple(target_side_length, input_shape[1]) / target_side_length

        if lcm_factor > 10:
            upsample_factor = 10
        else:
            upsample_factor = int(lcm_factor)

        img_upsampled = tf.image.resize(input_image, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                        size=2 * [upsample_factor * target_side_length])
        output_img = tf.nn.avg_pool2d(input=img_upsampled, ksize=[1, upsample_factor, upsample_factor, 1],
                                      strides=[1, upsample_factor, upsample_factor, 1], padding="VALID")
    return output_img
