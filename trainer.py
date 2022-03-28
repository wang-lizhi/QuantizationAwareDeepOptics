import json
import os
from datetime import datetime

import tensorflow as tf

from log import Logger


def prepare_image_dataset(task_name, dataset_name, dataset_loader_func_name, training_batch_size):
    from util.data.dataset_loader import DATASET_PATH
    root_dir = DATASET_PATH[dataset_name]
    train_dir = root_dir + "/train"
    val_dir = root_dir + "/validation"

    from importlib import import_module
    dataset_loader_func = getattr(import_module("util.data.dataset_loader"), dataset_loader_func_name)
    assert dataset_loader_func is not None, "Invalid dataset_loader_func_name."

    Logger.i("Preparing training datasets from {}...".format(train_dir))
    training_pairs = dataset_loader_func(train_dir,
                                         cache_name="./cache/%s-%s-train" % (task_name, dataset_name)) \
        .batch(training_batch_size).repeat().prefetch(tf.data.AUTOTUNE)

    Logger.i("Preparing validation datasets from {}...".format(val_dir))
    validation_pairs = dataset_loader_func(val_dir,
                                           cache_name="./cache/%s-%s-validation" % (task_name, dataset_name)) \
        .batch(training_batch_size).repeat()
    return training_pairs, validation_pairs


def print_args_summary(model_args, training_args):
    Logger.i("\n\n==============>Controlled Args<==============="
             "\n>>Model: \n", model_args, "\n >>Training:", training_args, )


def train(task_name, model, dataset_name, dataset_loader_func_name, loss, metrics, note,
          training_batch_size, step_per_epoch, total_epoch, checkpoint_monitor="mse", checkpoint_save_mode="min",
          summary_update_freq=30, validation_steps=None, exp_group_tag=None, controlled_model_args=None,
          controlled_training_args=None, pretrained_checkpoint_path_to_load=None, continue_training=False):
    assert model is not None, "The model must not be None."
    from summary import TensorBoardFix
    if continue_training:
        assert pretrained_checkpoint_path_to_load is not None, "Checkpoint file must be given to continue training."
        split_path = pretrained_checkpoint_path_to_load.split("/")
        train_log_dir_name = split_path[-3] + "/" + split_path[-2]
    else:
        train_log_dir_name = datetime.now().strftime("%Y%m%d-%H%M%S") + note
        if exp_group_tag is not None and exp_group_tag != "":
            Logger.i("Using group tag: ", exp_group_tag)
            train_log_dir_name = exp_group_tag + "/" + train_log_dir_name

    Logger.i("[DIR] training log directory name = ", train_log_dir_name)
    log_dir = './logs/' + train_log_dir_name

    # Dataset
    train_pairs, validation_pairs = prepare_image_dataset(task_name=task_name,
                                                          dataset_name=dataset_name,
                                                          training_batch_size=training_batch_size,
                                                          dataset_loader_func_name=dataset_loader_func_name)

    # <--Callbacks-->
    tensorboard_callback = TensorBoardFix(log_dir=log_dir, write_graph=False, write_images=False,
                                          update_freq=summary_update_freq, profile_batch=3)
    # Enable numerics check & performance profiler
    tf.summary.trace_on(graph=True, profiler=True)
    tf.summary.trace_export(name="quantization_aware_deep_optics_trace", step=0, profiler_outdir=log_dir)

    checkpoint_dir_path = "./checkpoint/" + train_log_dir_name
    checkpoint_file_path = "./checkpoint/" + train_log_dir_name + "/cp-{epoch:03d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file_path, verbose=1, save_best_only=False,
                                                     save_weights_only=True, save_freq=step_per_epoch*5,
                                                     mode=checkpoint_save_mode, monitor=checkpoint_monitor)
    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)
        Logger.i("Creating dir:", checkpoint_dir_path)
    if controlled_model_args is not None and not continue_training:
        with open(checkpoint_dir_path + "/controlled_model_args.json", 'a+') as model_args_json:
            json.dump(controlled_model_args, model_args_json)
    if controlled_training_args is not None and not continue_training:
        del controlled_training_args["metrics"]
        with open(checkpoint_dir_path + "/controlled_training_args.json", 'a+') as training_args_json:
            json.dump(controlled_training_args, training_args_json)

    # Loss & metrics
    from loss import LOSS_FUNCTION_FILTER
    loss = LOSS_FUNCTION_FILTER[loss]
    model.compile(loss=loss, metrics=metrics)
    model.build(input_shape=(training_batch_size,
                             controlled_model_args["image_patch_size"],
                             controlled_model_args["image_patch_size"],
                             controlled_model_args["input_channel_num"]))
    model.summary()
    print_args_summary(controlled_model_args, controlled_training_args)

    checkpoint_epoch = 0
    if pretrained_checkpoint_path_to_load is not None:
        Logger.i("Loading checkpoint file from: ", pretrained_checkpoint_path_to_load)
        checkpoint_epoch = int(pretrained_checkpoint_path_to_load[-8:-5])
        pretrained_checkpoint_to_load = tf.train.Checkpoint(model)
        pretrained_checkpoint_to_load.restore(pretrained_checkpoint_path_to_load)
        Logger.i("Restored checkpoint: ", pretrained_checkpoint_to_load)
        Logger.i("Start fitting from epoch ", checkpoint_epoch)

    model.fit(train_pairs, initial_epoch=checkpoint_epoch, epochs=total_epoch, validation_data=validation_pairs,
              validation_steps=validation_steps, verbose=1, steps_per_epoch=step_per_epoch,
              callbacks=[cp_callback, tensorboard_callback])
