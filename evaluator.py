import argparse
import json
import os
from importlib import import_module

import pandas as pd
import scipy.io as sio
import tensorflow as tf

import constants
from log import Logger
from metrics import ssim_metric, psnr_metric, psnr_hyper_metric, sam_metric, ergas_metric
from util.data.dataset_loader import DATASET_PATH

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('--checkpoint_dir', type=str, required=True,
                             help='checkpoint file directory')

argument_parser.add_argument('--tag_name', type=str, required=False, default=None,
                             help='checkpoint tag name.')

argument_parser.add_argument('--tag_vars', nargs='+', type=str, required=False, default=None,
                             help='Variables in tag (integers seperated by space), '
                                  'which will be fulfilled in %d position in <inner_tag>.')

argument_parser.add_argument('--as_training', required=False, default=False, action="store_true",
                             help='Evaluate the model in training model i.e. model call: [call(..., testing=False)]')

argument_parser.add_argument('--dataset', type=str, required=False, default=None,
                             help='Determine which dataset is used for testing. '
                                  'Default is read from the the `controlled_training_args` file of the checkpoint.')

argument_parser.add_argument('--real_data_dir', type=str, required=False, default=None,
                             help='Use real RGB data to reconstruct. '
                                  'This settings will overwrite the `dataset` option. '
                                  'The RGB data size should be 2048*2048.')

argument_parser.add_argument('--test_q', type=int, required=False, default=0,
                             help='Whether to apply the quantization during the test (for conventional DO only).')

arguments = argument_parser.parse_args()

EVAL_METRICS = {
    "MAE": lambda gt, pred: tf.reduce_mean(tf.abs(gt - pred)),
    "MSE": lambda gt, pred: tf.reduce_mean(tf.square(gt - pred)),
    "RMSE": lambda gt, pred: tf.sqrt(tf.reduce_mean(tf.square(gt - pred))),
    "SSIM": ssim_metric,
    "PSNR": psnr_metric,
    "PSNR-H": psnr_hyper_metric,
    "ERGAS": ergas_metric,
    "SAM": sam_metric
}

RESULT_EXPORT_DIR = "./eval-res/"


def crop_2016_to_512_patches(_img):
    flat_map_list = []
    for i in range(4):
        for j in range(4):
            flat_map_list.append(tf.image.crop_to_bounding_box(_img, (512 - 12) * i, (512 - 12) * j, 512, 512))
    return tf.data.Dataset.from_tensor_slices(flat_map_list)


def load_png_files_as_dataset_from_dir(_dir):
    rgb_png_path_list = tf.io.matching_files(os.path.join(_dir, '*.png'))
    from util.data.data_utils import decode_rgb_8bit_png
    png_ds = tf.data.Dataset.from_tensor_slices(rgb_png_path_list).map(decode_rgb_8bit_png) \
        .map(lambda _img: tf.pad(_img, paddings=[[16, 16], [16, 16], [0, 0]], mode="CONSTANT"))
    return png_ds


def evaluate(_model, checkpoint_inner_path, tag="default", checkpoint_parent_name=None, eval_dataset=None,
             with_images=True, with_mat=True, with_excel=True, with_gt=True,
             keep_in_training_mode=False, use_real_data_mode=False):
    _checkpoint = tf.train.Checkpoint(_model)
    _checkpoint_path_to_load = tf.train.latest_checkpoint(checkpoint_inner_path)
    Logger.i("Loading checkpoint file from: ", _checkpoint_path_to_load)
    _checkpoint.restore(_checkpoint_path_to_load)
    Logger.i("Ready to evaluate.")
    images = eval_dataset.as_numpy_iterator()
    i = 0

    metric_lists = {}
    list_excel_to_export = []

    from util.hyper_spec_drawer import save_hyper_to_tiled_rgb_file

    for hyper_img in images:
        Logger.i("Evaluating #%d" % i)

        hyper_img = tf.expand_dims(hyper_img, axis=0)
        predicted_image = _model(hyper_img, testing=not keep_in_training_mode)

        if with_images:
            save_hyper_to_tiled_rgb_file(predicted_image,
                                         RESULT_EXPORT_DIR + checkpoint_parent_name + "/" + tag + "/RGB-%d" % i,
                                         with_rgb=True)

            if not use_real_data_mode and with_gt:
                save_hyper_to_tiled_rgb_file(hyper_img,
                                             RESULT_EXPORT_DIR + checkpoint_parent_name + "/" + tag + "/GT-%d.png" % i,
                                             with_rgb=True)
        if with_mat:
            Logger.i("Saving MAT to: ", RESULT_EXPORT_DIR + checkpoint_parent_name + "/" + tag + "/MAT-Pred-%d.mat" % i)
            sio.savemat(
                RESULT_EXPORT_DIR + checkpoint_parent_name + "/" + tag + "/MAT-Pred-%d.mat" % i,
                {"pred": tf.squeeze(predicted_image, axis=0).numpy()},
                do_compression=False)
            if not use_real_data_mode and with_gt:
                sio.savemat(
                    RESULT_EXPORT_DIR + checkpoint_parent_name + "/" + tag + "/MAT-GT-%d.mat" % i,
                    {"gt": tf.squeeze(hyper_img, axis=0).numpy()},
                    do_compression=False)

        if use_real_data_mode:
            i += 1
            # Real data mode does not need full-image metrics calculation
            continue

        metrics_summary = []

        for metric_name, metric_func in EVAL_METRICS.items():
            metric_value = metric_func(hyper_img, predicted_image)
            Logger.i("> %s=" % metric_name, metric_value)
            if metric_name not in metric_lists:
                metric_lists[metric_name] = []

            metric_lists[metric_name].append(metric_value.numpy())

            metrics_summary.append(metric_value.numpy())

        list_excel_to_export.append(metrics_summary)
        i += 1

    Logger.i("===Evaluation Summary===\n", metric_lists)

    averages_to_return = []
    if use_real_data_mode:
        return averages_to_return

    for metric_name, _ in EVAL_METRICS.items():
        avg_value = tf.reduce_mean(metric_lists[metric_name])
        Logger.i(" > Avg %s=" % metric_name, avg_value)
        averages_to_return.append(str(avg_value.numpy()))

    del _checkpoint, _model

    if with_excel:
        data_frame = pd.DataFrame(list_excel_to_export, columns=EVAL_METRICS.keys())
        data_frame.to_csv(RESULT_EXPORT_DIR + checkpoint_parent_name + "/Excel_" + tag + ".csv")

    return averages_to_return


if __name__ == "__main__":
    eval_results = {}
    full_results = {}

    checkpoint_root_dir = "./checkpoint/"

    checkpoint_parent_dir = arguments.checkpoint_dir
    raw_tag_name = arguments.tag_name
    tag_vars = arguments.tag_vars
    as_training = arguments.as_training
    real_data_dir = arguments.real_data_dir
    test_q = arguments.test_q

    Logger.i("checkpoint_dir", checkpoint_parent_dir)
    Logger.i("tag_name", raw_tag_name)
    Logger.i("tag_vars", tag_vars)

    if tag_vars is None:
        tag_vars = [-1]

    for _var in tag_vars:
        if _var == -1:
            tag_name = raw_tag_name
        else:
            tag_name = raw_tag_name % _var

        if tag_name is None:
            checkpoint_child_dir = checkpoint_root_dir + checkpoint_parent_dir + '/'
            tag_name = "default_tag"
        else:
            checkpoint_child_dir = checkpoint_root_dir + checkpoint_parent_dir + '/' + tag_name

        Logger.i("Loading arguments from ", tag_name)

        with open(checkpoint_child_dir + "/controlled_model_args.json") as args_file:
            controlled_model_args = json.load(args_file)

        with open(checkpoint_child_dir + "/controlled_training_args.json") as args_file:
            controlled_training_args = json.load(args_file)

        ds_path = DATASET_PATH[controlled_training_args["dataset_name"]] + "/test"

        print("Loading datasets...")
        if real_data_dir is None:
            _dataset_name = controlled_training_args["dataset_name"]
            _dataset_loader_func_name = controlled_training_args["dataset_loader_func_name"]
            _task_name = controlled_training_args["task_name"]
            dataset_loader_func = getattr(
                import_module("util.data.dataset_loader"), _dataset_loader_func_name)
            dataset = dataset_loader_func(ds_path,
                                          cache_name="./cache/%s-%s-test" % (_task_name, _dataset_name))
            Logger.i(" > Data loader: ", _dataset_loader_func_name, "; Dataset: ", _dataset_name)

            assert dataset is not None, "No proper `dataset_loader_func_name` found in the controlled_training_args file of the checkpoint." \
                                        "The evaluator cannot know what dataset was used to train this checkpoint."
        else:
            controlled_model_args["reconstruction_network_args"]["input_size"] = (2048, 2048, 3)
            dataset = load_png_files_as_dataset_from_dir(real_data_dir)
            tag_name = "REAL-DATA-" + tag_name
            Logger.w("Using real data...")

        if test_q != 0:
            Logger.w("Quantization testing. Level=%d." % test_q)
            tag_name = tag_name + "-TestQ%dLv" % test_q
            controlled_model_args["doe_extra_args"] = {
                "quantization_level_cnt": test_q,
                "quantize_at_test_only": True,
                "adaptive_quantization": False,
                "alpha_blending": False
            }

        Logger.i("\n==controlled_model_args==\n", controlled_model_args)
        Logger.i("\n==controlled_training_args==\n", controlled_training_args)

        skip_optical_encoding = real_data_dir is not None

        Logger.i("Load as QDO...")
        from model import QuantizationAwareDeepOpticsModel

        if "Harvard" in controlled_training_args["dataset_name"]:
            Logger.w("Wavelength range changed to [420, 720] because of %s dataset."
                     % controlled_training_args["dataset_name"])
            model = QuantizationAwareDeepOpticsModel(**controlled_model_args,
                                                     wave_length_list=constants.wave_length_list_420_720nm,
                                                     skip_optical_encoding=skip_optical_encoding)
        else:
            Logger.i("Wavelength range changed to [400, 700] because of %s dataset."
                     % controlled_training_args["dataset_name"])
            model = QuantizationAwareDeepOpticsModel(**controlled_model_args,
                                                     skip_optical_encoding=skip_optical_encoding)

        if not skip_optical_encoding:
            if "CAVE" in controlled_training_args["dataset_name"]:
                model.build(input_shape=(1, 512, 512, 31))
            else:
                model.build(input_shape=(1, controlled_model_args["image_patch_size"],
                                         controlled_model_args["image_patch_size"],
                                         controlled_model_args["input_channel_num"]))
        else:
            model.build(input_shape=(1, 2048, 2048, 3))
        model.summary()
        Logger.i("Evaluating ", tag_name)

        if not os.path.exists(RESULT_EXPORT_DIR + checkpoint_parent_dir):
            os.makedirs(RESULT_EXPORT_DIR + checkpoint_parent_dir)
            Logger.i("Creating dir:", RESULT_EXPORT_DIR + checkpoint_parent_dir)
        if not os.path.exists(RESULT_EXPORT_DIR + checkpoint_parent_dir + "/" + tag_name):
            os.makedirs(RESULT_EXPORT_DIR + checkpoint_parent_dir + "/" + tag_name)
            Logger.i("Creating dir:", RESULT_EXPORT_DIR + checkpoint_parent_dir + "/" + tag_name)

        average_values = evaluate(model, checkpoint_inner_path=checkpoint_child_dir, tag=tag_name, eval_dataset=dataset,
                                  checkpoint_parent_name=checkpoint_parent_dir, keep_in_training_mode=as_training,
                                  use_real_data_mode=real_data_dir is not None)
        eval_results[tag_name] = "/".join(average_values)
        del checkpoint_child_dir
    Logger.i(json.dumps(eval_results))
