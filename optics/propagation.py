import numpy as np
import tensorflow as tf

from .util import transpose_2d_fft, transpose_2d_ifft, complex_exponent_tf


class FresnelPropagation(tf.keras.layers.Layer):
    def __init__(self, distance, discretization_size, wave_lengths, energy_penalty_mask_type=None):
        super().__init__()
        self.distance = distance
        self.wave_lengths = wave_lengths
        self.wave_nos = 2. * np.pi / wave_lengths
        self.discretization_size = discretization_size
        self.m_padding = None
        self.n_padding = None
        self.outside_sensor_boolean_mask = None
        self.energy_penalty_mask_type = energy_penalty_mask_type
        self.H = None

    def build(self, input_shape):
        batch_num, m_original, n_original, channel_num = input_shape
        # zero padding
        m_padding = m_original // 4
        n_padding = n_original // 4
        self.m_padding = m_padding
        self.n_padding = n_padding
        m_full = m_original + 2 * m_padding
        n_full = n_original + 2 * n_padding

        [x, y] = np.mgrid[-n_full // 2:n_full // 2, -m_full // 2:m_full // 2]

        # Spatial frequency
        fx = x / (self.discretization_size * n_full)  # max frequency = 1/(2*pixel_size)
        fy = y / (self.discretization_size * m_full)

        fx = tf.signal.ifftshift(fx)
        fy = tf.signal.ifftshift(fy)

        fx = fx[None, :, :, None]
        fy = fy[None, :, :, None]

        squared_sum = np.square(fx) + np.square(fy)

        constant_exponent_part = np.float64(self.wave_lengths * np.pi * -1. * squared_sum)
        self.H = complex_exponent_tf(self.distance * constant_exponent_part,
                                     dtype=tf.complex64, name='fresnel_kernel')

    def _propagate(self, input_field, training=None):
        padded_input_field = tf.pad(tensor=input_field,
                                    paddings=[[0, 0], [self.m_padding, self.m_padding],
                                              [self.n_padding, self.n_padding], [0, 0]])

        fourier_padded_input_field = transpose_2d_fft(padded_input_field)
        output_field = transpose_2d_ifft(fourier_padded_input_field * self.H)

        tf.summary.scalar(name="out_field_abs_sum", data=tf.reduce_sum(tf.cast(tf.abs(output_field), dtype=tf.float32)))

        return output_field[:, self.m_padding:-self.m_padding, self.n_padding:-self.n_padding, :]

    def call(self, inputs, training=None, **kwargs):
        return self._propagate(input_field=inputs, training=training)

