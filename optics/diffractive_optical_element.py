import math
from abc import ABC

import numpy as np
import tensorflow as tf

from log import Logger
from summary import image_normalization
from .util import complex_exponent_tf, transpose_2d_ifft, ifft_shift_2d_tf

BASE_PLANE_THICKNESS = 2 * 1e-3


def summary_height_map(height_map):
    tf.summary.image(name="HeightMapNormalized", data=image_normalization(
        tf.keras.activations.relu(height_map - BASE_PLANE_THICKNESS + 1.5e-6)))


def _phase_to_height_with_material_refractive_idx_func(_phase, _wavelength, _refractive_index_function):
    return _phase / (2 * math.pi / _wavelength) / (_refractive_index_function(_wavelength) - 1)


def _copy_quad_to_full(quad_map):
    height_map_half_left = tf.concat([tf.reverse(quad_map, axis=[0]), quad_map], axis=0)
    height_map_full = tf.concat([tf.reverse(height_map_half_left, axis=[1]), height_map_half_left], axis=1)
    return height_map_full


class DOELayer(ABC, tf.keras.layers.Layer):
    @staticmethod
    def shift_phase_according_to_height_map(input_field, height_map, wave_lengths, wavelength_to_refractive_index_func):
        """
        Calculates the phase shifts created by a height map with certain refractive index for light with specific wave length.

        Args:
            input_field: Input field.
            height_map: DOE height map.
            wave_lengths: Wavelength list.
            wavelength_to_refractive_index_func:  Refractive index function of the DOE material.

        Returns: Modulated wave field.
        """
        summary_height_map(height_map)
        delta_n = wavelength_to_refractive_index_func(wave_lengths) - 1
        wave_numbers = 2. * np.pi / wave_lengths
        wave_numbers = wave_numbers.reshape([1, 1, 1, -1])
        phi = wave_numbers * delta_n * height_map
        phase_shifts = complex_exponent_tf(phi)
        input_field = tf.cast(input_field, tf.complex64)
        shifted_field = tf.multiply(input_field, phase_shifts, name='phase_plate_shift')
        return shifted_field

    @staticmethod
    def add_height_map_noise(height_map, tolerance=None):
        if tolerance is not None:
            height_map = height_map + tf.random.uniform(shape=height_map.shape,
                                                        minval=-tolerance,
                                                        maxval=tolerance,
                                                        dtype=height_map.dtype)
            Logger.i("Simulated fabrication noise on height map: %0.2e" % tolerance)
        return height_map

    def preprocess_height_map(self, training=None, testing=None):
        return NotImplemented

    def modulate(self, input_field, preprocessed_height_map, height_map_regularizer, height_tolerance, wave_length_list,
                 wavelength_to_refractive_index_func):
        if height_map_regularizer is not None:
            self.add_loss(height_map_regularizer(preprocessed_height_map))
        preprocessed_height_map = self.add_height_map_noise(preprocessed_height_map, tolerance=height_tolerance)
        return self.shift_phase_according_to_height_map(
            input_field=input_field,
            height_map=preprocessed_height_map,
            wave_lengths=wave_length_list,
            wavelength_to_refractive_index_func=wavelength_to_refractive_index_func)


def read_pretrained_height_map(check_point_path):
    check_point = tf.train.latest_checkpoint(check_point_path)
    doe_var = tf.train.load_variable(check_point,
                                     "doe_layer/weight_height_map_radius_1d/.ATTRIBUTES/VARIABLE_VALUE")
    return doe_var


class FixedDOELayer(DOELayer):
    def __init__(self, wave_length_list, wavelength_to_refractive_index_func, height_map_initializer=None,
                 height_tolerance=None, quantization_levels=None, name="FixedDOELayer"):
        super(FixedDOELayer, self).__init__(name=name)
        self.wave_length_list = wave_length_list
        self.wavelength_to_refractive_index_func = wavelength_to_refractive_index_func
        self.height_map_initializer = height_map_initializer
        self.height_tolerance = height_tolerance
        self.weight_height_map = None
        self.height_map_shape = None
        self.quantization_levels = quantization_levels

    def build(self, input_shape):
        _, height, width, _ = input_shape
        self.height_map_shape = [1, height, width, 1]
        assert self.height_map_initializer is not None, "Height map must be given for `FixeDOELayer`."
        self.weight_height_map = self.add_weight(name="weight_height_map",
                                                 shape=self.height_map_shape,
                                                 dtype=tf.float32,
                                                 trainable=False,
                                                 initializer=self.height_map_initializer)

    def preprocess_height_map(self, training=None, testing=None):
        if self.quantization_levels is None:
            return self.weight_height_map
        else:
            Logger.i("The height map will be quantized to %d-level style." % self.quantization_levels)
            full_precision_value = self.weight_height_map
            _max_val = tf.reduce_max(full_precision_value)
            Logger.i("Height map max value={}".format(_max_val))
            quantized_value = tf.cast(tf.round((full_precision_value / _max_val) * (self.quantization_levels - 1)),
                                      dtype=tf.float32)
            quantized_value = (quantized_value / (self.quantization_levels - 1)) * _max_val
        return quantized_value

    def call(self, inputs, training=None, testing=None, **kwargs):
        return self.modulate(input_field=inputs,
                             preprocessed_height_map=self.preprocess_height_map(training=training, testing=testing),
                             height_map_regularizer=None,
                             height_tolerance=self.height_tolerance,
                             wave_length_list=self.wave_length_list,
                             wavelength_to_refractive_index_func=self.wavelength_to_refractive_index_func)


class HeightMapDOELayer(DOELayer):
    def __init__(self, wave_length_list, wavelength_to_refractive_index_func, block_size=1, height_map_initializer=None,
                 height_map_regularizer=None, height_tolerance=None, name="HeightMapDOELayer"):
        super(HeightMapDOELayer, self).__init__(name=name)
        self.wave_length_list = wave_length_list
        self.wavelength_to_refractive_index_func = wavelength_to_refractive_index_func
        self.block_size = block_size
        self.height_map_initializer = height_map_initializer
        self.height_map_regularizer = height_map_regularizer
        self.height_tolerance = height_tolerance
        self.weight_height_map = None
        self.height_map_shape = None
        self.height_constraint_max = _phase_to_height_with_material_refractive_idx_func(
            math.pi, 700 * 1e-9, wavelength_to_refractive_index_func)  # height_constraint_max

    def build(self, input_shape):
        _, height, width, _ = input_shape
        self.height_map_shape = [1, height // self.block_size, width // self.block_size, 1]
        if self.height_map_initializer is None:
            init_height_map_value = np.ones(shape=self.height_map_shape, dtype=np.float32) * 1e-4
            # init_height_map_value = read_pretrained_height_map()
            self.height_map_initializer = tf.keras.initializers.constant(value=init_height_map_value)
        self.weight_height_map = self.add_weight(name="weight_height_map_sqrt",
                                                 shape=self.height_map_shape,
                                                 dtype=tf.float32,
                                                 trainable=True,
                                                 initializer=self.height_map_initializer,
                                                 constraint=lambda x: tf.clip_by_value(x, -10, 10))

    def preprocess_height_map(self, training=None, testing=None):
        height_map = self.height_constraint_max * tf.sigmoid(self.weight_height_map)
        return height_map

    def call(self, inputs, training=None, testing=None, **kwargs):
        return self.modulate(input_field=inputs,
                             preprocessed_height_map=self.preprocess_height_map(training=training, testing=testing),
                             height_map_regularizer=self.height_map_regularizer,
                             height_tolerance=self.height_tolerance,
                             wave_length_list=self.wave_length_list,
                             wavelength_to_refractive_index_func=self.wavelength_to_refractive_index_func)


class QuantizedHeightMapDOELayer(DOELayer):
    def __init__(self, wave_length_list, wavelength_to_refractive_index_func, height_map_initializer=None,
                 height_map_regularizer=None, height_tolerance=None,
                 quantization_height_base_wavelength=700 * 1e-9, quantization_height_base_phase=2 * math.pi,
                 quantization_level_cnt=4, quantize_at_test_only=False,
                 adaptive_quantization=True, alpha_blending=True, step_per_epoch=960,
                 alpha_blending_start_epoch=5, alpha_blending_end_epoch=25,
                 name="QuantizedHeightMapDOELayer"):
        super(QuantizedHeightMapDOELayer, self).__init__(name=name)
        self.wave_length_list = wave_length_list
        self.wavelength_to_refractive_index_func = wavelength_to_refractive_index_func
        self.height_map_initializer = height_map_initializer
        self.height_map_regularizer = height_map_regularizer
        self.height_tolerance = height_tolerance
        self.step_counter = None
        self.weight_height_map = None
        self.height_map_shape = None
        self.alpha_blending = alpha_blending
        self.base_plane_thickness = BASE_PLANE_THICKNESS
        self.quantization_height_base_wavelength = quantization_height_base_wavelength
        self.quantization_base_height = _phase_to_height_with_material_refractive_idx_func(
            _phase=quantization_height_base_phase,
            _wavelength=quantization_height_base_wavelength,
            _refractive_index_function=self.wavelength_to_refractive_index_func)
        Logger.i("[DOE] Quantization base height: %.12e" % self.quantization_base_height)
        self.quantization_level_adaptive_weight = None
        self.quantization_level_cnt = quantization_level_cnt
        self.quantize_at_test_only = quantize_at_test_only
        self.adaptive_quantization = adaptive_quantization

        self.ALPHA_BLENDING_START_STEP = step_per_epoch * alpha_blending_start_epoch
        self.ALPHA_BLENDING_END_STEP = step_per_epoch * alpha_blending_end_epoch

        Logger.i("[QDO] Blending Start Step: %d" % self.ALPHA_BLENDING_START_STEP)
        Logger.i("[QDO] Blending End Step: %d" % self.ALPHA_BLENDING_END_STEP)
        Logger.i("[QDO] Blending Start Epoch: %d" % alpha_blending_start_epoch)
        Logger.i("[QDO] Blending End Epoch: %d" % alpha_blending_end_epoch)

    def generalizable_pre_build(self):
        # Initialize weights for alpha_blending of QDO or adaptive quantization of AQDO
        self.step_counter = self.add_weight(name="step_counter", shape=None, dtype=tf.int32, trainable=False,
                                            initializer=tf.keras.initializers.constant(value=0))
        if self.adaptive_quantization:
            self.quantization_level_adaptive_weight = self.add_weight(name="quantization_level_adaptive_weight",
                                                                      shape=self.quantization_level_cnt,
                                                                      trainable=True,
                                                                      initializer=tf.keras.initializers.constant(
                                                                          tf.zeros(shape=self.quantization_level_cnt)),
                                                                      constraint=(lambda x: tf.clip_by_value(x, -1, 1))
                                                                      if self.quantization_level_cnt > 2
                                                                      else (lambda x: tf.clip_by_value(x, -0.2, 0.2)))

        if self.height_map_initializer is None:
            init_height_map_value = np.ones(shape=self.height_map_shape, dtype=np.float32) * 1e-4
            self.height_map_initializer = tf.keras.initializers.constant(value=init_height_map_value)
            # Gaussian initializer
            # self.height_map_initializer = tf.random_normal_initializer(mean=0.0, stddev=1)

    def build(self, input_shape):
        _, height, width, _ = input_shape
        self.height_map_shape = [1, height, width, 1]
        self.generalizable_pre_build()
        self.weight_height_map = self.add_weight(name="weight_height_map",
                                                 shape=self.height_map_shape,
                                                 dtype=tf.float32,
                                                 trainable=True,
                                                 initializer=self.height_map_initializer,
                                                 constraint=lambda x: tf.clip_by_value(x, -1, 1)
                                                 )

    def preprocess_height_map(self, training=None, testing=None):

        @tf.custom_gradient
        def _round_keep_gradients(_x):
            def _grad(_dy):
                return _dy

            return tf.round(_x), _grad

        def _norm_to_0_and_1(_x):
            return (_x + 1) * 0.5  # tf.sigmoid(_x)

        def _full_precise_path(_weight):
            return _norm_to_0_and_1(_weight)

        def _quantized_path(_weight, _round_op, _quantization_level_cnt=4, _adaptive=False):
            _normed_height_map = _norm_to_0_and_1(_weight)
            quantized_levels = tf.cast(_round_op(_normed_height_map * (_quantization_level_cnt - 1)),
                                       dtype=tf.float32)  # 0, 1, 2, 3
            Logger.i("[QDO] Quantization levels：", _quantization_level_cnt)
            # ===== [AQDO] Adaptive quantization level tuning
            if _adaptive:
                Logger.i("[AQDO] Adaptive quantization-aware training enabled.")
                for level in range(1, _quantization_level_cnt):
                    level_fine_tune_weight = self.quantization_level_adaptive_weight[level]
                    quantized_levels = tf.where(tf.equal(quantized_levels, level),
                                                quantized_levels + 0.5 * level_fine_tune_weight,
                                                quantized_levels)
                    tf.summary.scalar(name="level_fine_tune_weight%d" % level, data=level_fine_tune_weight)
            # ===== ![AQDO] Adaptive quantization level tuning

            _quantized_height_map = quantized_levels / (_quantization_level_cnt - 1)  # 0-1

            # ===== [AQDO] Adaptive quantization level tuning (adaptive loss)
            if _adaptive and (testing is not True):
                adaptive_quantization_loss = tf.reduce_mean(tf.square(_quantized_height_map - _normed_height_map))
                tf.summary.scalar(name="adaptive_quantization_loss", data=adaptive_quantization_loss)
                self.add_loss(0.01 * adaptive_quantization_loss)  # fine-tune loss
            # ===== ![AQDO] Adaptive quantization level tuning (adaptive loss)

            return _quantized_height_map

        def _alpha_blending(_path1, _path2, _cur_step, _start_step, _end_step):
            if _cur_step < _start_step:
                # Full-precision
                quantization_blending_alpha = 0.0
            elif _cur_step > _end_step:
                # Full-quantized
                quantization_blending_alpha = 1.0
            else:
                # Quantization-aware training
                quantization_blending_alpha = tf.cast(
                    1.0 - ((_end_step - _cur_step) /
                           (_end_step - _start_step)) ** 3, dtype=tf.float32)

            tf.summary.scalar(name="quantization_blending_alpha", data=quantization_blending_alpha)

            return quantization_blending_alpha * _path1 + (1.0 - quantization_blending_alpha) * _path2

        def _base_plane_wrapper(_etching_height_map_weight):
            return self.base_plane_thickness - (self.quantization_base_height * _etching_height_map_weight)

        if self.height_map_regularizer is not None and (testing is not True):
            self.add_loss(self.height_map_regularizer(self.weight_height_map))

        if self.quantize_at_test_only:
            Logger.i("[DO] `quantize_at_test_only` mode is enabled."
                     "This mode can simulate the conventional deep optics "
                     "without considering the error caused by fabrication quantization.")
            assert not self.alpha_blending, "When `quantize_at_test_only` mode is enabled, " \
                                            "quantization-aware training option should not be True."
            assert not self.adaptive_quantization, "When `quantize_at_test_only` mode is enabled, " \
                                                   "adaptive quantization-aware training option should not be True."
            if testing is None:
                Logger.w("Argument `testing` is None in `quantize_at_test_only` mode.")
            if training is None:
                Logger.w("Argument `training` is None None in `quantize_at_test_only` mode.")
            if not training and testing:
                Logger.i("Testing in `quantize_at_test_only` mode, the DOE height map will be quantized.")

                final_processed_height_map = _base_plane_wrapper(
                    _quantized_path(self.weight_height_map,
                                    _round_op=tf.round,
                                    _quantization_level_cnt=self.quantization_level_cnt,
                                    _adaptive=False))
            else:
                Logger.i("Training in `quantize_at_test_only` mode, the DOE height map will be full-precision.")
                final_processed_height_map = _base_plane_wrapper(_full_precise_path(self.weight_height_map))
        else:
            Logger.i("[QDO] `quantization-aware` mode is enabled.")
            if self.alpha_blending:
                Logger.i("[QDO] using quantization-aware approach: <alpha_blending>.")
                # --- With alpha-blending

                final_processed_height_map = _base_plane_wrapper(_alpha_blending(
                    _path1=_quantized_path(
                        self.weight_height_map, _round_op=tf.round, _quantization_level_cnt=self.quantization_level_cnt,
                        _adaptive=self.adaptive_quantization),
                    _path2=_full_precise_path(self.weight_height_map),
                    _cur_step=self.step_counter,
                    _start_step=self.ALPHA_BLENDING_START_STEP,
                    _end_step=self.ALPHA_BLENDING_END_STEP))

            else:
                Logger.i("[QDO] using quantization-aware approach: <STE>.")
                # --- With STE
                final_processed_height_map = _base_plane_wrapper(_quantized_path(
                    _weight=self.weight_height_map, _round_op=_round_keep_gradients,
                    _quantization_level_cnt=self.quantization_level_cnt, _adaptive=False))

        # === QE Record
        ideal_cond = _base_plane_wrapper(_full_precise_path(self.weight_height_map))
        quantization_error_mae = tf.reduce_mean(tf.abs(ideal_cond - final_processed_height_map))
        quantization_error_mse = tf.reduce_mean(tf.square(ideal_cond - final_processed_height_map))
        tf.summary.scalar(name="quantization_error_mae", data=quantization_error_mae)
        tf.summary.scalar(name="quantization_error_mse", data=quantization_error_mse)
        # !=== QE Record

        tf.summary.scalar(name="step_counter", data=self.step_counter)
        if training:
            self.step_counter.assign_add(1)  # increase step

        return final_processed_height_map

    def call(self, inputs, training=None, testing=None, **kwargs):
        height_map = self.preprocess_height_map(training=training, testing=testing)
        return self.modulate(input_field=inputs,
                             preprocessed_height_map=height_map,
                             height_map_regularizer=None,
                             height_tolerance=self.height_tolerance,
                             wave_length_list=self.wave_length_list,
                             wavelength_to_refractive_index_func=self.wavelength_to_refractive_index_func)


class QuadSymmetricQuantizedHeightMapDoeLayer(QuantizedHeightMapDOELayer):

    def build(self, input_shape):
        _, height, width, _ = input_shape
        self.height_map_shape = [int(height / 2), int(width / 2)]
        self.generalizable_pre_build()
        self.weight_height_map = self.add_weight(name="weight_height_map_quad",
                                                 shape=self.height_map_shape,
                                                 dtype=tf.float32,
                                                 trainable=True,
                                                 initializer=self.height_map_initializer,
                                                 constraint=lambda x: tf.clip_by_value(x, -1, 1))

    def preprocess_height_map(self, training=None, testing=None):
        height_map_quad = super().preprocess_height_map(training=training, testing=testing)
        height_map_full = _copy_quad_to_full(height_map_quad)
        height_map_full = tf.expand_dims(height_map_full, axis=0)
        height_map_full = tf.expand_dims(height_map_full, axis=-1)  # reshape => [1, h, w, 1]
        return height_map_full


class RotationallySymmetricQuantizedHeightMapDOELayer(QuantizedHeightMapDOELayer):

    def build(self, input_shape):
        _, height, width, _ = input_shape
        self.height_map_shape = int(height / 2)
        self.generalizable_pre_build()

        self.weight_height_map = self.add_weight(name="weight_height_map_radius_1d",
                                                 shape=self.height_map_shape,
                                                 dtype=tf.float32,
                                                 trainable=True,
                                                 initializer=self.height_map_initializer,
                                                 constraint=lambda x: tf.clip_by_value(x, -1, 1))

    def preprocess_height_map(self, training=None, testing=None):
        height_map_1d = super().preprocess_height_map(training=training, testing=testing)

        radius = self.height_map_shape
        diameter = 2 * radius
        [x, y] = np.mgrid[0:diameter // 2, 0:diameter // 2].astype(np.float32)
        radius_distance = tf.sqrt(x ** 2 + y ** 2)

        height_map_quad = tf.where(tf.logical_and(tf.less(radius_distance, 1.0),
                                                  tf.less_equal(0.0, radius_distance)),
                                   height_map_1d[0], 0.0)
        for r in range(1, radius - 1):
            height_map_quad += tf.where(tf.logical_and(tf.less(radius_distance, tf.cast(r + 1, dtype=tf.float32)),
                                                       tf.less_equal(tf.cast(r, dtype=tf.float32), radius_distance)),
                                        height_map_1d[r], 0.0)

        height_map_full = _copy_quad_to_full(height_map_quad)

        height_map = tf.reshape(height_map_full, shape=[1, diameter, diameter, 1])

        return height_map


class FourierDOELayer(DOELayer):
    def __init__(self, wave_length_list, wavelength_to_refractive_index_func, frequency_range,
                 height_map_regularizer=None,
                 height_tolerance=None, name="FourierDOELayer"):
        super(FourierDOELayer, self).__init__(name=name)
        self.wave_length_list = wave_length_list
        self.wavelength_to_refractive_index_func = wavelength_to_refractive_index_func
        self.frequency_range = frequency_range
        self.height_map_initializer = tf.compat.v1.zeros_initializer()
        self.height_map_regularizer = height_map_regularizer
        self.height_tolerance = height_tolerance
        self.height_map_var = None
        self.height_map_full = None
        self.padding_width = None

        self.weight_fourier_real = None
        self.weight_fourier_imaginary = None

    def build(self, input_shape):
        assert self.frequency_range is not None, "Invalid args"
        _, height, width, _ = input_shape.as_list()
        frequency_range = self.frequency_range
        self.weight_fourier_real = self.add_weight('weight_fourier_coefficients_real',
                                                   shape=[1, int(height * frequency_range),
                                                          int(width * frequency_range), 1],
                                                   dtype=tf.float32, trainable=True,
                                                   initializer=self.height_map_initializer)
        self.weight_fourier_imaginary = self.add_weight('weight_fourier_coefficients_imaginary',
                                                        shape=[1, int(height * frequency_range),
                                                               int(width * frequency_range), 1],
                                                        dtype=tf.float32, trainable=True,
                                                        initializer=self.height_map_initializer)
        self.padding_width = int((1 - self.frequency_range) * height) // 2

    def preprocess_height_map(self, training=None, testing=None):
        fourier_coefficients = tf.complex(self.weight_fourier_real, self.weight_fourier_imaginary)
        fourier_coefficients_padded = tf.pad(tensor=fourier_coefficients,
                                             paddings=[[0, 0], [self.padding_width, self.padding_width],
                                                       [self.padding_width, self.padding_width], [0, 0]])
        height_map = tf.math.real(transpose_2d_ifft(ifft_shift_2d_tf(fourier_coefficients_padded)))
        return height_map

    def call(self, inputs, training=None, testing=None, **kwargs):
        return self.modulate(input_field=inputs,
                             preprocessed_height_map=self.preprocess_height_map(training=training, testing=testing),
                             height_map_regularizer=self.height_map_regularizer,
                             height_tolerance=self.height_tolerance,
                             wave_length_list=self.wave_length_list,
                             wavelength_to_refractive_index_func=self.wavelength_to_refractive_index_func)


class Rank1HeightMapDOELayer(DOELayer):
    def __init__(self, wave_length_list, wavelength_to_refractive_index_func, height_constraint_max,
                 height_map_regularizer=None, height_tolerance=None, name="Rank1ParameterizedHeightMapDOELayer"):
        super(Rank1HeightMapDOELayer, self).__init__(name=name)
        self.wave_length_list = wave_length_list
        self.wavelength_to_refractive_index_func = wavelength_to_refractive_index_func
        self.height_map_regularizer = height_map_regularizer
        self.height_tolerance = height_tolerance
        self.height_constraint_max = height_constraint_max

        self.weight_map_column = None
        self.weight_map_row = None

    def build(self, input_shape):
        _, height, width, _ = input_shape.as_list()
        column_shape = [1, width]
        row_shape = [height, 1]
        column_init_value = np.ones(shape=column_shape, dtype=np.float32) * 1e-2
        row_init_value = np.ones(shape=row_shape, dtype=np.float32) * 1e-2
        column_initializer = tf.keras.initializers.constant(value=column_init_value)
        row_initializer = tf.keras.initializers.constant(value=row_init_value)
        # Rank-1 parameterization: H = C * W
        self.weight_map_column = self.add_weight(name="weight_height_map_column",
                                                 shape=column_shape,
                                                 dtype=tf.float32,
                                                 trainable=True,
                                                 initializer=column_initializer)
        self.weight_map_row = self.add_weight(name="weight_height_map_row",
                                              shape=row_shape,
                                              dtype=tf.float32,
                                              trainable=True,
                                              initializer=row_initializer)

    def preprocess_height_map(self, training=None, testing=None):
        height_map_mul = tf.matmul(self.weight_map_row, self.weight_map_column)  # (h, w)
        height_map = 1.125 * 1e-6 * tf.sigmoid(height_map_mul)  # clip to [0, 1.125μm]
        height_map = tf.expand_dims(height_map, 0)  # (1, h, w)
        height_map = tf.expand_dims(height_map, -1)  # (1, h, w, 1)
        return height_map

    def call(self, inputs, training=None, testing=None, **kwargs):
        return self.modulate(input_field=inputs,
                             preprocessed_height_map=self.preprocess_height_map(training=training, testing=testing),
                             height_map_regularizer=self.height_map_regularizer,
                             height_tolerance=self.height_tolerance,
                             wave_length_list=self.wave_length_list,
                             wavelength_to_refractive_index_func=self.wavelength_to_refractive_index_func)
