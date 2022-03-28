import tensorflow as tf

import constants
from log import Logger


class QuantizationAwareDeepOpticsModel(tf.keras.Model):
    def get_config(self):
        config = super(QuantizationAwareDeepOpticsModel, self).get_config()
        config.update({
            "image_patch_size": self.image_patch_size,
            "sensor_distance": self.sensor_distance,
            "wavelength_to_refractive_index_func": self.wavelength_to_refractive_index_func,
            "wave_resolution": self.wave_resolution,
            "wave_length_list": self.wave_length_list,
            "sample_interval": self.sample_interval,
            "input_channel_num": self.input_channel_num,
            "doe_layer_type": self.doe_layer_type,
            "optical_system_optimizer": self.optical_system_optimizer,
            "reconstruction_network": self.reconstruction_network,
            "network_optimizer": self.network_optimizer,
            "srf_type": self.srf_type,
            "height_map_noise": self.height_map_noise
        })
        return config

    def __init__(self, image_patch_size, sensor_distance, wavelength_to_refractive_index_func_name, wave_resolution,
                 sample_interval, input_channel_num, doe_layer_type, depth_bin,
                 wave_length_list=constants.wave_length_list_400_700nm,
                 default_optimizer_learning_rate_args=None, reconstruction_network_type=None,
                 reconstruction_network_args=None, network_optimizer_learning_rate_args=None,
                 use_psf_fixed_camera=False,
                 srf_type=None, doe_extra_args=None, height_map_noise=None, skip_optical_encoding=False,
                 use_extra_optimizer=False, extra_optimizer_learning_rate_args=None, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        Logger.i("Wavelength list used：", wave_length_list)

        if doe_extra_args is None:
            doe_extra_args = {}

        self.image_patch_size = image_patch_size
        self.sensor_distance = sensor_distance

        self.wave_length_list = wave_length_list
        self.sample_interval = sample_interval
        self.wave_resolution = wave_resolution
        self.input_channel_num = input_channel_num

        self.doe_layer_type = doe_layer_type
        self.doe_layer = None

        if reconstruction_network_type == "res_block_u_net":
            from networks.res_block_u_net import get_res_block_u_net
            self.reconstruction_network = get_res_block_u_net(**reconstruction_network_args)

        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.optimizers.schedules import ExponentialDecay

        self.default_optimizer = None
        if default_optimizer_learning_rate_args is not None:
            self.default_optimizer = Adam(learning_rate=ExponentialDecay(**default_optimizer_learning_rate_args))

        self.network_optimizer = None
        if network_optimizer_learning_rate_args is not None:
            self.network_optimizer = Adam(learning_rate=ExponentialDecay(**network_optimizer_learning_rate_args))

        self.extra_optimizer = None
        if extra_optimizer_learning_rate_args is not None:
            self.extra_optimizer = Adam(learning_rate=ExponentialDecay(**extra_optimizer_learning_rate_args))

        self.srf_type = srf_type
        self._input_shape = None
        self.height_map_noise = height_map_noise

        self.skip_optical_encoding = skip_optical_encoding

        if not skip_optical_encoding:
            if not use_psf_fixed_camera:
                if use_extra_optimizer:
                    assert self.extra_optimizer is not None, "The `extra_optimizer` argument must be given " \
                                                             "when `use_extra_optimizer` is True."
                    self.train_step = self._train_step_uni_extra_optimizer
                    self.test_step = self._test_step
                else:
                    assert self.default_optimizer is not None, "The `default_optimizer` argument must be given " \
                                                               "when `use_psf_fixed_camera` is True."
                    if self.network_optimizer is not None:
                        assert self.reconstruction_network is not None, "The `network_optimizer` must not be None " \
                                                                        "when `reconstruction_network` is defined."
                    if self.network_optimizer is None:
                        self.train_step = self._train_step_uni_optimizer
                    else:
                        self.train_step = self._train_step_separated_optimizer
                self.test_step = self._test_step

                from constants import MATERIAL_REFRACTIVE_INDEX_FUNCS
                self.wavelength_to_refractive_index_func = \
                    MATERIAL_REFRACTIVE_INDEX_FUNCS[wavelength_to_refractive_index_func_name]
                assert self.wavelength_to_refractive_index_func is not None, \
                    "Unsupported `doe_material` argument. It should be in: " + str(
                        MATERIAL_REFRACTIVE_INDEX_FUNCS.keys())

                doe_general_args = {
                    "wave_length_list": wave_length_list,
                    "wavelength_to_refractive_index_func": self.wavelength_to_refractive_index_func,
                    "height_map_initializer": None,
                    "height_tolerance": height_map_noise,
                }

                Logger.i("\n\n==============>DOE Args<===============\n  > General:\n", doe_general_args,
                         "\n  > Extra:\n", doe_extra_args, "==============<DOE Args>===============\n\n")

                if doe_layer_type == "rank1":
                    from optics.diffractive_optical_element import Rank1HeightMapDOELayer
                    self.doe_layer = Rank1HeightMapDOELayer(**doe_general_args,
                                                            height_map_regularizer=None,
                                                            **doe_extra_args)
                elif doe_layer_type == 'htmp':
                    from optics.diffractive_optical_element import HeightMapDOELayer
                    from optics.util import laplace_l1_regularizer
                    self.doe_layer = HeightMapDOELayer(**doe_general_args,
                                                       height_map_regularizer=laplace_l1_regularizer(scale=0.1),
                                                       **doe_extra_args)
                elif doe_layer_type == 'htmp-quant':
                    from optics.diffractive_optical_element import QuantizedHeightMapDOELayer
                    from optics.util import laplace_l1_regularizer
                    self.doe_layer = QuantizedHeightMapDOELayer(
                        **doe_general_args,
                        **doe_extra_args)
                elif doe_layer_type == 'htmp-quant-quad':
                    from optics.diffractive_optical_element import QuadSymmetricQuantizedHeightMapDoeLayer
                    from optics.util import laplace_l1_regularizer
                    self.doe_layer = QuadSymmetricQuantizedHeightMapDoeLayer(
                        **doe_general_args,
                        **doe_extra_args)
                elif doe_layer_type == 'htmp-quant-sym':
                    from optics.diffractive_optical_element import RotationallySymmetricQuantizedHeightMapDOELayer
                    from optics.util import laplace_l1_regularizer
                    self.doe_layer = RotationallySymmetricQuantizedHeightMapDOELayer(
                        **doe_general_args,
                        **doe_extra_args)

                assert self.doe_layer is not None, "Problems occurred and the DOE layer is None. Check your settings."

                from optics.camera import Camera
                from optics.sensor import Sensor

                sensor = None
                if srf_type is not None:
                    sensor = Sensor(srf_type=srf_type)

                self.optical_system = Camera(wave_resolution=self.wave_resolution,
                                             wave_length_list=self.wave_length_list,
                                             sensor_distance=self.sensor_distance,
                                             sensor_resolution=(self.image_patch_size, self.image_patch_size),
                                             sensor=sensor,
                                             input_sample_interval=self.sample_interval,
                                             doe_layer=self.doe_layer,
                                             input_channel_num=self.input_channel_num,
                                             depth_list=depth_bin, should_use_planar_incidence=False,
                                             should_depth_dependent=False).done()
            else:
                Logger.w("Using PSFFixedCamera as optical system layer.")
                assert self.extra_optimizer is not None, "The `extra_optimizer` must not be None " \
                                                         "when using PSFFixedCamera."
                self.train_step = self._train_step_uni_extra_optimizer
                self.test_step = self._test_step
                from optics.camera import PSFFixedCamera
                self.optical_system = PSFFixedCamera(wave_resolution=wave_resolution,
                                                     sensor_resolution=(self.image_patch_size, self.image_patch_size))
        else:
            Logger.w("The optical system layer is disabled。")
            Logger.w("Using `extra_optimizer` as the optimizer.")
            self.train_step = self._train_step_uni_extra_optimizer
            self.optical_system = None
        self.model_description = "DOE{}_SpItv{}_SsDst{}_ScDst{}_WvRes{}_ImgSz{}_SRF{}" \
            .format(doe_layer_type, sample_interval, sensor_distance, depth_bin[0], wave_resolution[0],
                    image_patch_size, srf_type)

    def _test_step(self, data):
        x = data
        y = x
        y_pred = self(x, training=False, testing=False)
        # Updates stateful loss metrics.
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def _train_step_uni_optimizer(self, data):
        input_image = data
        ground_truth = input_image
        with tf.GradientTape() as tape:
            predicted_image = self(input_image, training=True, testing=False)
            training_loss = self.compiled_loss(ground_truth, predicted_image)
            regularization_loss = tf.math.reduce_sum(self.losses)
            total_loss = training_loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.default_optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(ground_truth, predicted_image)
        tf.summary.scalar(name="total_loss", data=total_loss)
        tf.summary.scalar(name="training_loss", data=training_loss)
        tf.summary.scalar(name="regularization_loss", data=regularization_loss)
        return {metric.name: metric.result() for metric in self.metrics}

    def _train_step_uni_extra_optimizer(self, data):
        input_image = data
        ground_truth = input_image
        with tf.GradientTape() as tape:
            predicted_image = self(input_image, training=True, testing=False)
            training_loss = self.compiled_loss(ground_truth, predicted_image)
            regularization_loss = tf.math.reduce_sum(self.losses)
            total_loss = training_loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.extra_optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(ground_truth, predicted_image)
        tf.summary.scalar(name="total_loss", data=total_loss)
        tf.summary.scalar(name="training_loss", data=training_loss)
        tf.summary.scalar(name="regularization_loss", data=regularization_loss)
        return {metric.name: metric.result() for metric in self.metrics}

    def _train_step_separated_optimizer(self, data):
        input_image = data
        ground_truth = input_image

        # The gradient should be persistent when using separated optimizers
        with tf.GradientTape(persistent=True) as tape:
            predicted_image = self(input_image, training=True, testing=False)
            training_loss = self.compiled_loss(ground_truth, predicted_image)
            regularization_loss = tf.reduce_sum(self.losses)
            total_loss = training_loss + regularization_loss

        # Separated optimizer: optical system part
        optical_system_gradients = tape.gradient(total_loss, self.optical_system.trainable_variables)
        self.default_optimizer.apply_gradients(
            zip(optical_system_gradients, self.optical_system.trainable_variables))
        # Separated optimizer: reconstruction network part
        network_gradients = tape.gradient(total_loss, self.reconstruction_network.trainable_variables)
        self.network_optimizer.apply_gradients(
            zip(network_gradients, self.reconstruction_network.trainable_variables))

        self.compiled_metrics.update_state(ground_truth, predicted_image)
        tf.summary.scalar(name="total_loss", data=total_loss)
        tf.summary.scalar(name="training_loss", data=training_loss)
        tf.summary.scalar(name="regularization_loss", data=regularization_loss)
        return {metric.name: metric.result() for metric in self.metrics}

    def call(self, inputs, training=None, testing=None, **kwargs):
        if not self.skip_optical_encoding:
            x = self.optical_system(inputs, training=training, testing=testing)
        else:
            x = inputs
        tf.summary.image(name="SensorImage", data=x, max_outputs=1)
        if self.reconstruction_network is not None:
            reconstructed = self.reconstruction_network(x)
            if training:
                from summary import summary_hyper_spec_image
                summary_hyper_spec_image(reconstructed, name="ReconstructedImage")
            return reconstructed
        else:
            return x
