import tensorflow as tf

from log import Logger
from optics.sensor_srfs import SRF_MAP


class Sensor(tf.keras.layers.Layer):
    def __init__(self, srf_type):
        super(Sensor, self).__init__()
        self.srf_type = srf_type
        self.srf = SRF_MAP[srf_type]
        assert self.srf is not None, "Sensor SRF type must be given." \
                                     "Supported mode: gray, rgb, rggb1, and rggb4."
        Logger.i("[Sensor] SRF Type=", srf_type)

    def call(self, inputs, **kwargs):
        sensor_image = self.srf(inputs)
        return sensor_image

    def get_config(self):
        config = super(Sensor, self).get_config()
        config.update({"srf_type": self.srf_type})
        return config
