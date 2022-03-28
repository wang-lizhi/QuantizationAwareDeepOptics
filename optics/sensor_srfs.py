import tensorflow as tf

from constants import SRF_GS3_U3_41S4C_BGR_31_CHANNEL_400_700NM, \
    SRF_GS3_U3_41S4C_BGR_31_CHANNEL_420_720NM, SRF_SAPPHIRE_S25A30CL_GRAY_31_CHANNEL_400_700NM, \
    SRF_MOCK_BGR_31_CHANNEL_400_700NM

SRF_BGR_31_CHANNEL_400_700_NM = SRF_GS3_U3_41S4C_BGR_31_CHANNEL_400_700NM
SRF_BGR_31_CHANNEL_420_720_NM = SRF_GS3_U3_41S4C_BGR_31_CHANNEL_420_720NM
SRF_GRAY_31_CHANNEL_400_700_NM = SRF_SAPPHIRE_S25A30CL_GRAY_31_CHANNEL_400_700NM


def broadcast_2d_height_width_tensor_to_4d_with_single_batch_and_channel(tensor_2d):
    return tf.expand_dims(tf.expand_dims(tensor_2d, axis=0), axis=-1)


def mask_and_reshape(tensor, mask, half_height, half_width):
    tensor = tf.boolean_mask(tensor, mask)
    tensor = tf.reshape(tensor, [half_height, half_width, 1, -1])  # h, w, 1, b
    return tensor


def inverse_simulated_gray_channel_camera_spectral_response_function(input_gray_image):
    channel_num = 31
    gray_channel_expanded = tf.repeat(input_gray_image, repeats=channel_num, axis=-1)
    gray_channel_expanded = gray_channel_expanded * tf.reduce_sum(SRF_GRAY_31_CHANNEL_400_700_NM) / SRF_GRAY_31_CHANNEL_400_700_NM
    return gray_channel_expanded


def inverse_simulated_rgb_channel_camera_spectral_response_function(input_rgb_image,
                                                                    input_as_rggb=False):
    channel_num = 31
    masked_response_function = SRF_BGR_31_CHANNEL_400_700_NM
    green_2_channel = None
    wave_length_sum_response = tf.reduce_sum(masked_response_function, axis=0)
    if input_as_rggb:
        red_channel, green_channel, green_2_channel, blue_channel = tf.split(input_rgb_image, num_or_size_splits=4,
                                                                             axis=-1)
    else:
        red_channel, green_channel, blue_channel = tf.split(input_rgb_image, num_or_size_splits=3, axis=-1)

    red_channel_expanded = tf.repeat(red_channel, repeats=channel_num, axis=-1)
    green_channel_expanded = tf.repeat(green_channel, repeats=channel_num, axis=-1)
    blue_channel_expanded = tf.repeat(blue_channel, repeats=channel_num, axis=-1)

    red_channel_expanded = red_channel_expanded * (masked_response_function[2] / wave_length_sum_response)
    green_channel_expanded = green_channel_expanded * (masked_response_function[1] / wave_length_sum_response)
    blue_channel_expanded = blue_channel_expanded * (masked_response_function[0] / wave_length_sum_response)

    if input_as_rggb:
        assert green_2_channel is not None
        green_2_channel_expanded = tf.repeat(green_2_channel, repeats=channel_num, axis=-1) \
                                   * (masked_response_function[1] / wave_length_sum_response)
        green_channel_expanded = (green_2_channel_expanded + green_channel_expanded) / 2

    return red_channel_expanded + green_channel_expanded + blue_channel_expanded


def simulated_gray_camara_spectral_response_function(hyper_spectral_image):
    gray_responses = SRF_GRAY_31_CHANNEL_400_700_NM * hyper_spectral_image
    return tf.reduce_sum(gray_responses, axis=-1, keepdims=True) / tf.reduce_sum(SRF_GRAY_31_CHANNEL_400_700_NM)


def simulated_rgb_camera_spectral_response_function(hyper_spectral_image):
    channel_num = 31
    masked_response_function = SRF_BGR_31_CHANNEL_400_700_NM
    # Red
    red_response = hyper_spectral_image * tf.reshape(masked_response_function[2], shape=[1, 1, 1, channel_num])
    red_channel = tf.reduce_sum(red_response, axis=-1) / tf.reduce_sum(masked_response_function[2])
    # Green
    green_response = hyper_spectral_image * tf.reshape(masked_response_function[1], shape=[1, 1, 1, channel_num])
    green_channel = tf.reduce_sum(green_response, axis=-1) / tf.reduce_sum(masked_response_function[1])
    # Blue
    blue_response = hyper_spectral_image * tf.reshape(masked_response_function[0], shape=[1, 1, 1, channel_num])
    blue_channel = tf.reduce_sum(blue_response, axis=-1) / tf.reduce_sum(masked_response_function[0])
    # Stack RGB channels
    rgb_image = tf.stack([red_channel, green_channel, blue_channel], axis=-1)
    # Shape=(batch, height, width, 3)
    return rgb_image


def simulated_rgb_camera_spectral_response_function_for_visualization(hyper_spectral_image):
    channel_num = 31
    masked_response_function = SRF_MOCK_BGR_31_CHANNEL_400_700NM
    red_response = hyper_spectral_image * tf.reshape(masked_response_function[2], shape=[1, 1, 1, channel_num])
    red_channel = tf.reduce_sum(red_response, axis=-1) / tf.reduce_sum(masked_response_function[2])
    green_response = hyper_spectral_image * tf.reshape(masked_response_function[1], shape=[1, 1, 1, channel_num])
    green_channel = tf.reduce_sum(green_response, axis=-1) / tf.reduce_sum(masked_response_function[1])
    blue_response = hyper_spectral_image * tf.reshape(masked_response_function[0], shape=[1, 1, 1, channel_num])
    blue_channel = tf.reduce_sum(blue_response, axis=-1) / tf.reduce_sum(masked_response_function[0])
    rgb_image = tf.stack([red_channel, green_channel, blue_channel], axis=-1)
    # Shape=(batch, height, width, 3)
    return rgb_image


def simulated_rggb_bayer_raw_1_channel_response_function(hyper_spectral_image):
    rgb_image = simulated_rgb_camera_spectral_response_function(hyper_spectral_image)
    _, height, _, _ = rgb_image.shape
    red, green, blue = tf.split(rgb_image, num_or_size_splits=3, axis=-1)  # b, h, w, 1
    from optics.sensor_bayer_pattern import RGGB_BAYER_PATTERN_MASK
    # Combine RGGB channels into a single channel using the mosaicking pattern
    bc = broadcast_2d_height_width_tensor_to_4d_with_single_batch_and_channel
    masked_r = red * bc(RGGB_BAYER_PATTERN_MASK[height].r)
    masked_g1 = green * bc(RGGB_BAYER_PATTERN_MASK[height].g1)
    masked_g2 = green * bc(RGGB_BAYER_PATTERN_MASK[height].g2)
    masked_b = blue * bc(RGGB_BAYER_PATTERN_MASK[height].b)
    summed_rggb_raw = masked_r + masked_g1 + masked_g2 + masked_b
    # Shape=(batch, height, width, 1)
    return summed_rggb_raw


def simulated_rggb_bayer_with_malvar_2004_demosaicing_3_channel_response_function(hyper_spectral_image):
    from optics.demosaicing import demosaicing_bayer_malvar_2004_rggb
    return demosaicing_bayer_malvar_2004_rggb(
        simulated_rggb_bayer_raw_1_channel_response_function(hyper_spectral_image))


def simulated_rggb_bayer_separated_4_channel_response_function(hyper_spectral_image):
    rgb_image = simulated_rgb_camera_spectral_response_function(hyper_spectral_image)
    _, height, width, _ = hyper_spectral_image.shape  # b, h, w, c
    rgb_image = tf.transpose(rgb_image, perm=[1, 2, 3, 0])  # b, h, w, c => h, w, c, b
    red, green, blue = tf.split(rgb_image, num_or_size_splits=3, axis=2)  # h, w, 1, b
    from optics.sensor_bayer_pattern import RGGB_BAYER_PATTERN_BOOLEAN_MASK
    half_width = int(width / 2)
    half_height = int(height / 2)
    r = mask_and_reshape(red, RGGB_BAYER_PATTERN_BOOLEAN_MASK[height].r, half_height, half_width)  # h, w, 1, b
    g1 = mask_and_reshape(green, RGGB_BAYER_PATTERN_BOOLEAN_MASK[height].g1, half_height, half_width)
    g2 = mask_and_reshape(green, RGGB_BAYER_PATTERN_BOOLEAN_MASK[height].g2, half_height, half_width)
    b = mask_and_reshape(blue, RGGB_BAYER_PATTERN_BOOLEAN_MASK[height].b, half_height, half_width)
    combined_4_channel_image = tf.concat([r, g1, g2, b], axis=2)  # h, w, 4, b
    combined_4_channel_image = tf.transpose(combined_4_channel_image, perm=[3, 0, 1, 2])
    # Shape=(batch, height/2, width/2, 4)
    return combined_4_channel_image


SRF_MAP = {
    'gray': simulated_gray_camara_spectral_response_function,
    'rgb': simulated_rgb_camera_spectral_response_function,
    'rggb1': simulated_rggb_bayer_raw_1_channel_response_function,
    'rggb4': simulated_rggb_bayer_separated_4_channel_response_function,
    'rggb-demosaic': simulated_rggb_bayer_with_malvar_2004_demosaicing_3_channel_response_function
}

SRF_OUTPUT_SIZE_LAMBDA = {
    "rggb-demosaic": lambda p: (p, p, 3),
    "rggb4:": lambda p: (p / 2, p / 2, 4),
    "rggb1": lambda p: (p, p, 1),
    "rgb": lambda p: (p, p, 3),
    "rgb-keep-lum-diff": lambda p: (p, p, 3),
    "gray": lambda p: (p, p, 1)
}
