import tensorflow as tf

from optics.sensor_srfs import simulated_rgb_camera_spectral_response_function
from util.data.file_io import save_numpy_to_file


def wavelength_to_rgb(wavelength, gamma=0.8):
    """Convertor function
     Args:
         wavelength: A given wavelength of light to an approximate RGB color value. The wavelength must be given in nanometers in the range from 380 nm through 750 nm.
         gamma: The gamma value.

    return: R, G, B values. Maximum is 1.
    """

    wavelength = float(wavelength)
    if 380 <= wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif 440 <= wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif 490 <= wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif 510 <= wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif 580 <= wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif 645 <= wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    result = R, G, B
    return result


def gray_image_to_color(gray_image, r, g, b):
    r_channel = gray_image * r
    g_channel = gray_image * g
    b_channel = gray_image * b
    return tf.concat([r_channel, g_channel, b_channel], axis=-1)


def tile_hyper_channels_to_rgb_space_patch(hyper_spec_image, row=5, col=7, padding=0, norm=False):
    channel_count = hyper_spec_image.shape[-1]
    width = hyper_spec_image.shape[-2]
    height = hyper_spec_image.shape[-3]
    single_channels = tf.split(hyper_spec_image, num_or_size_splits=channel_count, axis=-1)
    tiled_image = None

    blank = tf.ones(shape=[height + padding * 2, width + padding * 2, 3], dtype=float)
    for i in range(row):
        concat_list = []
        for j in range(col):
            cur_index = i * col + j
            if cur_index < channel_count:
                _r, _g, _b = wavelength_to_rgb(400 + 10 * cur_index)
                _single_channel = single_channels[cur_index]
                if norm:
                    _single_channel /= tf.reduce_max(_single_channel)
                patch_to_append = gray_image_to_color(_single_channel, _r, _g, _b)
                patch_to_append = tf.pad(patch_to_append,
                                         paddings=[[padding, padding], [padding, padding], [0, 0]],
                                         mode="CONSTANT", constant_values=1.0)
                concat_list.append(patch_to_append)
            else:
                concat_list.append(blank)
        cur_row = tf.concat(concat_list, axis=-2)
        if i == 0:
            tiled_image = cur_row
        else:
            tiled_image = tf.concat([tiled_image, cur_row], axis=-3)

    return tiled_image


def save_hyper_to_tiled_rgb_file(hyper_image, save_path, with_hyper_tiles=False, with_rgb=True, norm=False):
    hyper_image /= tf.reduce_max(hyper_image)
    if with_hyper_tiles:
        tiled = tile_hyper_channels_to_rgb_space_patch(hyper_image, padding=3, norm=norm)
        tiled = tf.cast(tiled * 255, dtype=tf.uint8)
        tiled = tf.image.encode_png(tiled)
        print("Saving tiled hyperspectral images...")
        save_numpy_to_file(save_path + "-hyper.png", tiled.numpy())
    if with_rgb:
        rgb = simulated_rgb_camera_spectral_response_function(hyper_image)
        rgb = rgb / tf.reduce_max(rgb)
        rgb = tf.squeeze(tf.cast(rgb * 255, dtype=tf.uint8), axis=0)
        rgb = tf.image.encode_png(rgb)
        print("Saving RGB images...")
        save_numpy_to_file(save_path + "-rgb.png", rgb.numpy())

