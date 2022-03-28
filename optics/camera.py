import enum

import numpy as np
import scipy.io as sio
import tensorflow as tf

from log import Logger
from summary import summary_hyper_spec_image, image_normalization
from .diffractive_optical_element import DOELayer
from .noise import gaussian_noise
from .propagation import FresnelPropagation
from .sensor import Sensor
from .util import depth_dep_convolution, image_convolve_with_psf, complex_exponent_tf


class PSFMode(enum.Enum):
    FULL_PAD_IMAGE = 0
    FULL_PAD_IMAGE_WITH_SENSOR_LEAK_REG = 1
    FULL_RESIZE = 2
    CROP_VALID = 3


class Camera(tf.keras.layers.Layer):
    """
    Camera simulation class
    """

    def __init__(self, wave_resolution, wave_length_list, sensor_distance, sensor_resolution, input_sample_interval,
                 sensor: Sensor = None, doe_layer: DOELayer = None, input_channel_num=31,
                 noise_model=gaussian_noise, depth_map=None, depth_list=None, otfs=None,
                 should_use_planar_incidence=False, should_depth_dependent=False,
                 psf_mode=PSFMode.FULL_RESIZE,
                 name="camera"):
        """
        Initialize a Camera object.

        Args:
            wave_resolution: The wave field resolution, which is identical to the resolution of the DOE and its PSF.
            wave_length_list: Wave lengths that used in simulation. Default is 400nm to 700nm with 10nm intervals.
            sensor_distance: The distance between the DOE and the sensor plane.
            sensor_resolution: The resolution of the sensor, which is also the image size of the output.
            input_sample_interval: The sample interval, which is the pixel pitch of the DOE.
            doe_layer: The DOE layer object instance. It should be an instance of class `DOELayer`.
            input_channel_num: The number of input channel. Default is 31.
            noise_model: the noise model used in sensor.
            depth_map: (Optional) The depth map for depth-variant PSF generation.
            depth_list: (Optional) The depth index for depth-variant tasks.
            otfs: (Optional) Pre-built OTF.
            should_use_planar_incidence: If use planar light. This option will overwrite the depth-variant options.
            should_depth_dependent: If the task is depth-variant. This option will be overwrite by `should_use_planar_incidence`.
            name: The model name.
        """
        super(Camera, self).__init__(name=name)
        self.noise_model = noise_model
        self.wave_resolution = wave_resolution
        self.wave_length_list = wave_length_list
        self.sensor_resolution = sensor_resolution

        self.input_sample_interval = input_sample_interval
        self.sensor_distance = sensor_distance
        self.sensor = sensor
        self.otfs = otfs
        self.psfs = None
        self.target_psf = None
        self.input_channel_num = input_channel_num

        self.psf_mode = psf_mode
        should_resize_image_to_psf_size = True

        if psf_mode == PSFMode.FULL_RESIZE:
            should_resize_image_to_psf_size = True
            Logger.w("The PSF resize mode is enabled.")

        if should_resize_image_to_psf_size \
                and self.wave_resolution[0] == self.sensor_resolution[0] \
                and self.wave_resolution[1] == self.sensor_resolution[1]:
            should_resize_image_to_psf_size = False
            Logger.w("The Image resizing is disabled "
                     "because the `wave_resolution` and `sensor_resolution` is identical.")

        if should_resize_image_to_psf_size and self.psf_mode == PSFMode.CROP_VALID:
            should_resize_image_to_psf_size = False
            Logger.w("The Image resizing is disabled "
                     "because the PSF mode has been set to `CROP_VALID`.")
        if should_resize_image_to_psf_size and \
                (self.psf_mode == PSFMode.FULL_PAD_IMAGE or
                 self.psf_mode == PSFMode.FULL_PAD_IMAGE_WITH_SENSOR_LEAK_REG):
            should_resize_image_to_psf_size = False
            Logger.w("The Image resizing is disabled "
                     "because the PSF mode has been set to `FULL_PAD_IMAGE`.")

        self.flag_resize_image_to_psf_size = should_resize_image_to_psf_size

        # Incidence
        self.simulated_incidence = None
        self.flag_use_planar_incidence = should_use_planar_incidence
        self.physical_size = float(self.wave_resolution[0] * self.input_sample_interval)
        Logger.i("DOE physical size = %0.2e m.\n Wave resolution = %d." % (self.physical_size, self.wave_resolution[0]))
        self.pixel_size = self.input_sample_interval * np.array(wave_resolution) / np.array(sensor_resolution)

        # Propagation operator (from DOE to sensor)
        self.propagation = FresnelPropagation(distance=self.sensor_distance,
                                              discretization_size=self.input_sample_interval,
                                              wave_lengths=self.wave_length_list)

        self.invalid_energy_mask = None
        # PSF masks
        if psf_mode == PSFMode.CROP_VALID or psf_mode == PSFMode.FULL_PAD_IMAGE_WITH_SENSOR_LEAK_REG:
            from util.mask_generator import circle_mask
            self.invalid_energy_mask = circle_mask(full_square_length=wave_resolution[0],
                                                   inner_circle_radius=sensor_resolution[0] // 2)

        # DOE
        self.doe_layer = doe_layer

        # Depth-dependent settings
        if should_depth_dependent and should_use_planar_incidence:
            should_depth_dependent = False
            Logger.w("Option `should_depth_dependent` has been overwritten to False "
                     "because the `should_use_planar_incidence` is True.")
        elif should_depth_dependent and depth_list is None:
            should_depth_dependent = False
            Logger.w("Option `should_depth_dependent` has been overwritten to False "
                     "because the `depth_list` is not given.")
        self.flag_depth_dependent = should_depth_dependent

        self.depth_map = depth_map
        self.depth_list = depth_list

        self.noise_sigma = 0.001
        self._input_shape = None

    def build(self, input_shape):
        self._input_shape = input_shape

    def done(self):
        """
        Compile the camera. The Camera objection must be called done() before it is called as a `keras.layers.Layer` object.

        Returns: Camera object done.
        """
        if self.flag_use_planar_incidence:
            if self.depth_list is not None:
                Logger.w("It's invalid to set `depth_list` when using planar incidence.")
            self.simulated_incidence = planar_light(self.wave_resolution)
        else:
            assert self.depth_list is not None, "There should be at least one element in `depth_list` " \
                                                "when using spherical wave incidence filed."
            self.simulated_incidence = point_source_of_light_spherical_wave_field(depth_list=self.depth_list,
                                                                                  physical_size=self.physical_size,
                                                                                  wave_resolution=self.wave_resolution,
                                                                                  wave_lengths=self.wave_length_list)
        return self

    def generate_psf_from(self, input_fields, training=True, testing=None):
        assert self.doe_layer is not None, "The `doe_layer` is None. Call `attach()` method to add one."
        psfs = []
        for depth_idx, input_field in enumerate(input_fields):
            # Modulate with the DOE.
            field_after_height_map = self.doe_layer(input_field, training=training, testing=testing)
            field = circular_aperture(field_after_height_map)
            # Propagate to sensor plane.
            sensor_incident_field = self.propagation(field)
            psf = get_intensities(sensor_incident_field)
            # PSF energy mask. Mask will introduce a regularization loss on the PSF.
            if self.invalid_energy_mask is not None:
                psf = tf.math.divide(psf, tf.reduce_sum(input_tensor=psf, axis=[1, 2], keepdims=True),
                                     name='psf_before_cropping_depth_idx_%d' % depth_idx)
                psf_invalid_energy = psf * self.invalid_energy_mask
                summary_hyper_spec_image(image_normalization(psf_invalid_energy),
                                         name='PSF%d-InvalidEnergy' % depth_idx,
                                         with_single_channel=False, norm_channel=True)
                psf_invalid_energy = tf.reduce_sum(psf_invalid_energy)
                tf.summary.scalar(name="psf_invalid_energy_mean", data=psf_invalid_energy)
                self.add_loss(psf_invalid_energy)

            if self.psf_mode == PSFMode.CROP_VALID:
                crop_length = (self.wave_resolution[0] - self.sensor_resolution[0]) // 2
                psf = psf[:, crop_length:-crop_length, crop_length:-crop_length, :]
                tf.debugging.assert_equal(psf.shape[1], self.sensor_resolution[0],
                                          message="Cropped PSF height must be identical to the target image height "
                                                  "in CROP_VALID mode.")
                tf.debugging.assert_equal(psf.shape[2], self.sensor_resolution[1],
                                          message="Cropped PSF width must be identical to the target image width "
                                                  "in CROP_VALID mode.")
            # Keep the energy sum of each channel to 1
            psf = tf.math.divide(psf, tf.reduce_sum(input_tensor=psf, axis=[1, 2], keepdims=True),
                                 name='psf_depth_idx_%d' % depth_idx)
            # Record the  PSF
            summary_hyper_spec_image(image_normalization(psf), name='PSFNormed%d' % depth_idx,
                                     with_single_channel=True, norm_channel=True)

            transposed_psf = tf.transpose(a=psf, perm=[1, 2, 0, 3], name="transposed_psf_%d" % depth_idx)
            # Transpose to (height, width, batch=1, channels)
            psfs.append(transposed_psf)
        return psfs

    def get_pre_sensor_image(self, input_img, psfs):
        depth_map = self.depth_map

        if self.flag_depth_dependent:
            sensor_img = depth_dep_convolution(input_img, psfs, disc_depth_map=depth_map)
        else:
            sensor_img = image_convolve_with_psf(input_img, psfs[0], otf=self.otfs, img_shape=self._input_shape)
        sensor_img = tf.cast(sensor_img, tf.float32)
        return sensor_img

    def call(self, inputs, training=None, testing=None, **kwargs):
        """
        Args:
            inputs: Input image tensor.
            training: If under training.
            testing: If under testing.

        Returns: The sensor image with resolution of `sensor_resolution`.
        """
        assert self.simulated_incidence is not None, "Camera incidence is None. " \
                                                     "Call `done()` of the Camera instance before training/inference."
        if self.sensor is None:
            Logger.w("No `Sensor` object in camera.")
        psfs = self.generate_psf_from(input_fields=self.simulated_incidence, training=training, testing=testing)

        if self.flag_resize_image_to_psf_size:
            # Size assertion
            tf.debugging.assert_none_equal(inputs.shape[1], self.wave_resolution[0],
                                           message="Argument `flag_do_up_sampling` should not be True."
                                                   "Because the PSF and the input image already have same size.")
            inputs = tf.image.resize(inputs, self.wave_resolution,
                                     method=tf.image.ResizeMethod.BILINEAR)
            tf.debugging.assert_equal(inputs.shape[1], self.wave_resolution[0],
                                      message="Unexpected results."
                                              "The PSF and the input image should have same size after resizing.")

        pre_sensor_image = self.get_pre_sensor_image(input_img=inputs, psfs=psfs)

        noise_sigma = self.noise_sigma
        if training and noise_sigma is not None:
            pre_sensor_image = self.noise_model(image=pre_sensor_image, std_dev=noise_sigma)
            Logger.i("Sensor noise (Gaussian): stddev=%0.2e" % noise_sigma)

        if self.sensor is not None:
            sensor_image = self.sensor(pre_sensor_image)
        else:
            sensor_image = pre_sensor_image

        if self.flag_resize_image_to_psf_size:
            # Resize to sensor resolution.
            sensor_image = tf.image.resize(sensor_image, self.sensor_resolution, method=tf.image.ResizeMethod.BILINEAR)

        if self.sensor is not None:
            tf.summary.image(name="SensorImage", data=sensor_image, max_outputs=1)

        return sensor_image

    def get_config(self):
        config = super(Camera, self).get_config()
        config.update({
            "wave_resolution": self.wave_resolution,
            "wave_length_list": self.wave_length_list,
            "sensor_distance": self.sensor_distance,
            "sensor_resolution": self.sensor_resolution,
            "input_sample_interval": self.input_sample_interval,
            "doe_layer": self.doe_layer,
            "input_channel_num": self.input_channel_num,
            "noise_model": self.noise_model,
            "depth_map": self.depth_map,
            "depth_bin": self.depth_list,
            "otfs": self.otfs,
            "should_use_planar_incidence": self.flag_use_planar_incidence,
            "should_do_up_sampling": self.flag_resize_image_to_psf_size,
            "should_depth_dependent": self.flag_depth_dependent,
            "name": "camera"})
        return config



class PSFFixedCamera(tf.keras.layers.Layer):
    def __init__(self, wave_resolution=(1024, 1024), sensor_resolution=(512, 512)):
        super(PSFFixedCamera, self).__init__()
        self._input_shape = None
        self.psf = None
        self.flag_resize_image_to_psf_size = False
        self.sensor_resolution = sensor_resolution
        self.wave_resolution = wave_resolution
        self.response_curve = None

    def build(self, input_shape):
        self._input_shape = input_shape
        self.psf = self.add_weight(name="PSF", shape=(1, 1024, 1024, 31),
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(
                                       value=sio.loadmat(
                                           "./exported-mat-files/"
                                           "20211115-212655-psf-PSF-"
                                           "SK1300-210923-20nmNoise-4Lv-AdaABN0-Sd50-Sc1m-1EluBND7.mat")["psf"]),
                                   trainable=False)
        self.psf = tf.transpose(self.psf, perm=[1, 2, 0, 3])  # h, w, 1, 31

    def call(self, inputs, training=None, testing=None, **kwargs):

        if self.flag_resize_image_to_psf_size:
            inputs = tf.image.resize(inputs, self.wave_resolution,
                                     method=tf.image.ResizeMethod.BILINEAR)
        hyper_sensor_img = image_convolve_with_psf(inputs, self.psf, otf=None, img_shape=self._input_shape)

        from optics.sensor_srfs import simulated_rgb_camera_spectral_response_function
        sensor_img = simulated_rgb_camera_spectral_response_function(hyper_sensor_img)
        if training:
            sensor_img = gaussian_noise(image=sensor_img, std_dev=0.001)

        if self.flag_resize_image_to_psf_size:
            sensor_img = tf.image.resize(sensor_img, self.sensor_resolution, method=tf.image.ResizeMethod.BILINEAR)

        summary_hyper_spec_image(image=tf.transpose(self.psf, perm=[2, 0, 1, 3]), name="PSF",
                                 with_single_channel=True, norm_channel=True)
        return sensor_img


def circular_aperture(input_field):
    input_shape = input_field.shape.as_list()
    [x, y] = np.mgrid[-input_shape[1] // 2: input_shape[1] // 2, -input_shape[2] // 2: input_shape[2] // 2].astype(
        np.float64)

    max_val = np.amax(x)

    r = np.sqrt(x ** 2 + y ** 2)[None, :, :, None]
    aperture = (r < max_val).astype(np.float64)
    return aperture * input_field


def planar_light(wave_resolution):
    return [tf.ones(wave_resolution, dtype=tf.float32)[None, :, :, None]]


def point_source_of_light_spherical_wave_field(depth_list, physical_size, wave_resolution, wave_lengths,
                                               point_location=(0, 0)):
    distances = depth_list
    if distances is None:
        distances = []
    wave_res_n, wave_res_m = wave_resolution
    [x, y] = np.mgrid[-wave_res_n // 2:wave_res_n // 2, -wave_res_m // 2:wave_res_m // 2].astype(np.float64)
    x = x / wave_res_n * physical_size
    y = y / wave_res_m * physical_size

    x0, y0 = point_location

    squared_sum = (x - x0) ** 2 + (y - y0) ** 2
    wave_nos = 2. * np.pi / wave_lengths
    wave_nos = wave_nos.reshape([1, 1, 1, -1])

    input_fields = []
    for distance in distances:
        curvature = tf.sqrt(squared_sum + tf.cast(distance, tf.float64) ** 2)
        curvature = tf.expand_dims(tf.expand_dims(curvature, 0), -1)
        spherical_wavefront = complex_exponent_tf(wave_nos * curvature, dtype=tf.complex64)
        input_fields.append(spherical_wavefront)
    return input_fields


def get_intensities(input_field):
    return tf.square(tf.abs(input_field), name='intensities')
