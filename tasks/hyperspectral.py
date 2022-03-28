from metrics import ssim_metric, psnr_metric, sam_metric, psnr_hyper_metric, ergas_metric
from model import QuantizationAwareDeepOpticsModel
from optics.sensor_srfs import SRF_OUTPUT_SIZE_LAMBDA


upsample_rate = 2
image_patch_size = 512
doe_resolution = 512 * upsample_rate
doe_layer_type = "htmp-quant-sym"
srf_type = "rgb"
network_input_size = SRF_OUTPUT_SIZE_LAMBDA[srf_type](image_patch_size)
batch_size = 4
step_per_epoch = 1672 // batch_size

controlled_training_args = {
    "task_name": "HS", "dataset_name": "ICVL512-MAT", "loss": 'mae',
    "metrics": [ssim_metric, psnr_metric, psnr_hyper_metric, sam_metric, ergas_metric],
    "checkpoint_monitor": "psnr_hyper_metric", "checkpoint_save_mode": "max",
    "training_batch_size": batch_size, "step_per_epoch": step_per_epoch, "total_epoch": 50,
    "summary_update_freq": step_per_epoch // 4,
    "dataset_loader_func_name": "load_icvl_full_mat_512"
}

controlled_model_args = {
    "image_patch_size": image_patch_size, "sensor_distance": 50e-3,
    "wavelength_to_refractive_index_func_name": None,
    "sample_interval": 8e-6 / upsample_rate,
    "wave_resolution": (doe_resolution, doe_resolution),
    "input_channel_num": 31, "depth_bin": [1],
    "doe_layer_type": doe_layer_type,
    "srf_type": srf_type,
    "default_optimizer_learning_rate_args": {
        "initial_learning_rate": 0.01, "decay_steps": 500, "decay_rate": 0.8, "name": "default_opt_lr"},
    "network_optimizer_learning_rate_args": {
        "initial_learning_rate": 0.001, "decay_steps": 500, "decay_rate": 0.8, "name": "network_opt_lr"},
    "reconstruction_network_type": "res_block_u_net",
    "reconstruction_network_args": {
        "filter_root": 32, "depth": 7, "output_channel": 31, "input_size": network_input_size,
        "activation": 'elu', "batch_norm": True, "batch_norm_after_activation": False,
        "final_activation": 'sigmoid', "net_num": 1, "extra_upsampling": (srf_type == "rggb4"),
        "remove_first_long_connection": False, "channel_attention": False,
        "kernel_initializer": 'he_uniform', "final_kernel_initializer": 'glorot_uniform'
    },
    "height_map_noise": None
}


def train(doe_material="BK7", with_doe_noise=True, quantization_level=256, quantize_at_test_only=False,
          alpha_blending=False, adaptive_quantization=False, checkpoint=None, continue_training=False,
          tag=None, sensor_distance_mm=30, scene_depth_m=5):

    note = "-" + doe_material
    note += "-20nmNoise" if with_doe_noise else "-NoNoise"
    note += "-" + str(quantization_level) + "Lv"

    if quantize_at_test_only:
        assert not alpha_blending, "When `quantize_at_test_only` is True, `alpha_blending` cannot be set to True."
        assert not adaptive_quantization, \
            "when `quantize_at_test_only` is True, it's invalid to set enable `adaptive_quantization`."
        note += "-FullTrain"
    else:
        note += ("-AdaAB" if adaptive_quantization else "-NoAda-AB") if alpha_blending else "-STE"

    controlled_model_args["wavelength_to_refractive_index_func_name"] = doe_material
    controlled_model_args["height_map_noise"] = 20e-9 if with_doe_noise else None

    controlled_model_args["sensor_distance"] = sensor_distance_mm * 1e-3
    note += "-Sd%d" % sensor_distance_mm

    controlled_model_args["depth_bin"] = [scene_depth_m]
    note += "-Sc%dm" % scene_depth_m

    controlled_model_args["doe_extra_args"] = {
        "quantization_level_cnt": quantization_level,
        "quantize_at_test_only": quantize_at_test_only,
        "adaptive_quantization": adaptive_quantization,
        "alpha_blending": alpha_blending,
        "step_per_epoch": step_per_epoch,
        "alpha_blending_start_epoch": 5,
        "alpha_blending_end_epoch": 40
    }

    import trainer
    trainer.train(
        note=note,
        **controlled_training_args,
        exp_group_tag=tag,
        model=QuantizationAwareDeepOpticsModel(**controlled_model_args),
        controlled_model_args=controlled_model_args,
        controlled_training_args=controlled_training_args,
        pretrained_checkpoint_path_to_load=checkpoint,
        continue_training=continue_training
    )
