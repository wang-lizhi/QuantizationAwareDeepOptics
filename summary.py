import tensorflow as tf

from metrics import ssim_metric, psnr_metric, psnr_hyper_metric, post_recheck_ssim_metric, \
    post_recheck_psnr_hyper_metric, post_recheck_psnr_metric, unpack_reconstruction_from_prediction, \
    unpack_element_from_prediction
from optics.sensor_srfs import simulated_rgb_camera_spectral_response_function


class TensorBoardFix(tf.keras.callbacks.TensorBoard):
    """
    This fixes incorrect step values when using the TensorBoard callback with custom summary ops
    """

    def on_train_begin(self, *args, **kwargs):
        super(TensorBoardFix, self).on_train_begin(*args, **kwargs)
        tf.summary.experimental.set_step(self._train_step)

    def on_test_begin(self, *args, **kwargs):
        super(TensorBoardFix, self).on_test_begin(*args, **kwargs)
        tf.summary.experimental.set_step(self._train_step)  # _val_step


class EpochSummaryCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, image_evaluation_dataset, step_per_epoch):
        super().__init__()
        self.log_dir = log_dir
        self.image_evaluation_dataset = image_evaluation_dataset
        self.global_epoch = 0
        self.step_per_epoch = step_per_epoch
        self.writer = tf.summary.create_file_writer(logdir=self.log_dir)

    def set_model(self, model):
        self.model = model

    def on_train_end(self, logs=None):
        self.writer.close()

    @staticmethod
    def calculate_metrics(ground_truth, predicted_image, summary_id=0, post_recheck=False):
        if post_recheck:
            ssim = post_recheck_ssim_metric(ground_truth, predicted_image)
            psnr = post_recheck_psnr_metric(ground_truth, predicted_image)
            psnr_hyper = post_recheck_psnr_hyper_metric(ground_truth, predicted_image)
            summary_hyper_spec_image(image=unpack_reconstruction_from_prediction(predicted_image), name="%d-PredictedImage" % summary_id)
            summary_hyper_spec_image(image=unpack_element_from_prediction(predicted_image, pos=1), name="%d-PostReEncode" % summary_id)
            summary_hyper_spec_image(image=unpack_element_from_prediction(predicted_image, pos=2), name="%d-PreSensor" % summary_id)
        else:
            ssim = ssim_metric(ground_truth, predicted_image)
            psnr = psnr_metric(ground_truth, predicted_image)
            psnr_hyper = psnr_hyper_metric(ground_truth, predicted_image)
            summary_hyper_spec_image(image=predicted_image, name="%d-PredictedImage" % summary_id)
        return ssim, psnr, psnr_hyper

    def on_epoch_end(self, epoch, logs=None):
        self.global_epoch += 1  # the first epoch is number 1
        with tf.name_scope("EpochEndTest") as scope:
            i = 1
            test_ssim_sum = 0.0
            test_psnr_sum = 0.0
            test_psnr_hyper_sum = 0.0
            for input_image in self.image_evaluation_dataset:
                ground_truth = input_image
                tf.print("[Test] Epoch %d: Testing pair %d...\n" % (self.global_epoch, i))
                predicted_image = self.model(input_image, training=False, testing=True)
                # Write GT in the first epoch only.
                if epoch == 1:
                    summary_hyper_spec_image(ground_truth, name="%d-GroundTruth" % i)

                ssim, psnr, psnr_hyper = self.calculate_metrics(ground_truth, predicted_image, summary_id=i)

                tf.summary.scalar("%d-TestSSIM" % i, ssim)
                tf.summary.scalar("%d-TestPSNR" % i, psnr)
                tf.summary.scalar("%d-TestPSNRHyper" % i, psnr_hyper)
                tf.print("[Test] Epoch %d:[%d] SSIM=%f; PSNR=%f; PSNR_Hyper=%f" % (self.global_epoch,
                                                                                   i, ssim, psnr, psnr_hyper))
                test_ssim_sum += ssim
                test_psnr_sum += psnr
                test_psnr_hyper_sum += psnr_hyper
                i += 1
            avg_ssim = test_ssim_sum / (i - 1)
            avg_psnr = test_psnr_sum / (i - 1)
            avg_psnr_hyper = test_psnr_hyper_sum / (i - 1)
            tf.print("[Test] Epoch %d: Avg. SSIM=%f; PSNR=%f; PSNR_Hyper=%f" % (self.global_epoch,
                                                                                avg_ssim, avg_psnr, avg_psnr_hyper))
            tf.summary.scalar("0-AvgTestSSIM", avg_ssim)
            tf.summary.scalar("0-AvgTestPSNR", avg_psnr)
            tf.summary.scalar("0-AvgTestPSNRHyper", avg_psnr_hyper)


def image_normalization(image, given_max=None, given_min=None):
    """
    Normalize input images
    Args:
        image: image tensor
        given_max: maximum value given. If None, the result of tf.reduce_max(image) will be used.
        given_min: minimum value given. If None, the result of tf.reduce_min(image) will be used.

    Returns: Normalized image
    """
    if given_min is None:
        given_min = tf.math.reduce_min(image)
    image -= given_min

    if given_max is None:
        given_max = tf.math.reduce_max(image)
    if given_max != 0:
        image /= given_max
    return image


def summary_hyper_spec_image(image, name="InputImage", description=None, with_single_channel=False, norm_all=True,
                             norm_channel=False):
    with tf.name_scope(name):
        rgb_image = simulated_rgb_camera_spectral_response_function(image)
        if norm_all:
            image = image_normalization(image)
        tf.summary.image(name="%s-RGB" % name, data=rgb_image, description=description, max_outputs=1)
        if with_single_channel:
            summary_image_by_channel(image, name, norm_channel=norm_channel)


def summary_image_by_channel(image, name="ImageByChannel", norm_channel=False):
    input_channel_num = image.shape[-1]
    input_split = tf.split(image, input_channel_num, axis=-1, name="input_split_channels")
    i = 0
    for single_channel_input in input_split:
        desc = None
        if norm_channel:
            single_channel_input = image_normalization(single_channel_input)
        tf.summary.image(name='%s-channel%d' % (name, i), data=single_channel_input, description=desc, max_outputs=1)
        i += 1
