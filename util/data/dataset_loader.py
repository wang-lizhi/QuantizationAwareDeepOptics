import os

from log import Logger
from util.data.data_utils import *

DATASET_PATH = {
    "ICVL512-MAT": "/your/path/to/datasets/ICVL",
}


# ************************************************************************
# *******************************  CAVE  *********************************
# ************************************************************************

def load_cave_datasets(dataset_dir, cache_name="cache", verbose=True):
    cave = []
    for root, dirs, files in os.walk(dataset_dir):
        for _dir in dirs:
            hyper_png_path_list = tf.io.matching_files(os.path.join(root, _dir, '*.png'))
            png_cnt = 0
            for _ in hyper_png_path_list:
                png_cnt += 1
            if png_cnt != 0:
                if "CAVE" in root:
                    if verbose:
                        Logger.i("[Dataset] Loading from <", root, "/", _dir, ">\t (", png_cnt, "\t PNG files)")
                    cave.append(hyper_png_path_list)
                else:
                    if verbose:
                        Logger.w("[Dataset] Ignoring non-CAVE dataset")
            else:
                if verbose:
                    Logger.i("[Dataset] Skipping empty directory: ", root, _dir)

    def set_patch_size(_x):
        _x.set_shape((512, 512, 31))
        return _x

    cave_dataset = tf.data.Dataset.from_tensor_slices(cave) \
        .map(decode_hyper_image_from_png_list) \
        .map(set_patch_size).cache(cache_name + "-cave")
    return cave_dataset


def load_cave_datasets_with_rotated_256_patches(dataset_dir, cache_name="cache", verbose=True):
    cave_dataset = load_cave_datasets(dataset_dir, cache_name, verbose)
    cave_dataset = cave_dataset.flat_map(crop_cave_to_5_patches).flat_map(flat_map_image_self_and_transpose)
    return cave_dataset


def load_cave_hs_256(dataset_dir, cache_name="cache", verbose=False):
    cave_dataset = load_cave_datasets(dataset_dir, cache_name, verbose)
    cave_dataset = cave_dataset.flat_map(crop_cave_to_5_patches)
    return cave_dataset


# ************************************************************************
# *******************************  ICVL  *********************************
# ************************************************************************

def load_icvl_datasets(dataset_dir, cache_name="cache", verbose=False):
    icvl = []
    for root, dirs, files in os.walk(dataset_dir):
        for _dir in dirs:
            hyper_png_path_list = tf.io.matching_files(os.path.join(root, _dir, '*.png'))
            png_cnt = 0
            for _ in hyper_png_path_list:
                png_cnt += 1
            if png_cnt != 0:
                if verbose:
                    Logger.i("Loading from <", root, "/", _dir, ">\t (", png_cnt, "\t PNG files)")
                if "ICVL" in root:
                    if verbose:
                        Logger.i("[Dataset] Loaded as ICVL png files.")
                    icvl.append(hyper_png_path_list)
                else:
                    if verbose:
                        Logger.w("[Dataset] Ignoring non-ICVL files.")
            else:
                if verbose:
                    Logger.i("[Dataset] Skipping empty directory: ", root, _dir)
    icvl_dataset = tf.data.Dataset.from_tensor_slices(icvl) \
        .map(decode_hyper_image_from_png_list).cache(cache_name + "-icvl")
    return icvl_dataset


def load_icvl_hs_256(dataset_dir, cache_name="cache", verbose=False):
    icvl_dataset = load_icvl_datasets(dataset_dir, cache_name, verbose)
    icvl_dataset = icvl_dataset.flat_map(flat_map_256_patches_from_dataset)
    return icvl_dataset


def load_icvl_hs_512(dataset_dir, cache_name="cache", verbose=False):
    icvl_dataset = load_icvl_datasets(dataset_dir, cache_name, verbose)
    icvl_dataset = icvl_dataset.flat_map(flat_map_512_patches_from_dataset)
    return icvl_dataset


def flat_map_512_overlapped_patches_from_ICVL_dataset(_img):
    flat_map_operation_list = [tf.image.resize(images=_img, size=[512, 512], method=ResizeMethod.BILINEAR)]
    third_height = 464
    third_width = 434
    for i in range(0, 1392, third_height):
        for j in range(0, 1300, third_width):
            flat_map_operation_list.append(safe_crop_to_bounding_box(_img, i, j, 512, 512))
    return tf.data.Dataset.from_tensor_slices(flat_map_operation_list)


def flat_map_256_overlapped_patches_from_ICVL_dataset(_img):
    flat_map_operation_list = [tf.image.resize(images=_img, size=[256, 256], method=ResizeMethod.BILINEAR)]
    for i in range(0, 1392, 232):
        for j in range(0, 1300, 260):
            flat_map_operation_list.append(safe_crop_to_bounding_box(_img, i, j, 256, 256))
    return tf.data.Dataset.from_tensor_slices(flat_map_operation_list)


def flat_map_768_overlapped_patches_from_ICVL_dataset(_img):
    flat_map_operation_list = [tf.image.resize(images=_img, size=[768, 768], method=ResizeMethod.BILINEAR),
                               safe_crop_to_bounding_box(_img, 0, 0, 768, 768),
                               safe_crop_to_bounding_box(_img, 624, 0, 768, 768),
                               safe_crop_to_bounding_box(_img, 0, 532, 768, 768),
                               safe_crop_to_bounding_box(_img, 624, 532, 768, 768)
                               ]
    return tf.data.Dataset.from_tensor_slices(flat_map_operation_list)


def load_icvl_full_mat(dataset_dir, verbose=False):
    import h5py

    def icvl_mat_generator(_mat_list):
        for _mat_path in _mat_list:
            hyper = h5py.File(_mat_path)["rad"]
            if verbose:
                tf.print("Decoding ICVL MAT: <shape=", hyper.shape, ">@", _mat_path)
            yield hyper

    if verbose:
        Logger.i("Dataset loaded from: ", dataset_dir)
    mat_list = tf.io.matching_files(str(dataset_dir) + "/*.mat")
    mat_dataset = tf.data.Dataset \
        .from_generator(icvl_mat_generator, args=[mat_list],
                        output_types=tf.float32, output_shapes=(31, None, None)) \
        .map(lambda _img_to_norm: _img_to_norm / 4095.0) \
        .map(lambda _img_to_transpose: tf.transpose(_img_to_transpose, perm=[1, 2, 0]))
    return mat_dataset


def load_icvl_full_mat_512(dataset_dir, cache_name, verbose=True):
    return load_icvl_full_mat(dataset_dir, verbose=verbose).flat_map(
        flat_map_512_overlapped_patches_from_ICVL_dataset).cache(cache_name)


def load_icvl_full_mat_256(dataset_dir, cache_name, verbose=True):
    return load_icvl_full_mat(dataset_dir, verbose=verbose).flat_map(
        flat_map_256_overlapped_patches_from_ICVL_dataset).cache(cache_name)


def load_icvl_full_mat_512_norm(dataset_dir, cache_name, verbose=True):
    return load_icvl_full_mat_512(dataset_dir, cache_name, verbose).map(lambda _x: _x / tf.reduce_max(_x))


def load_icvl_full_mat_512_shuffle(dataset_dir, cache_name, verbose=True):
    return load_icvl_full_mat_512(dataset_dir, cache_name, verbose)


def load_icvl_full_mat_768(dataset_dir, cache_name, verbose=True):
    return load_icvl_full_mat(dataset_dir, verbose=verbose).flat_map(
        flat_map_768_overlapped_patches_from_ICVL_dataset).cache(cache_name)


# ************************************************************************
# ******************************  Harvard  *******************************
# ************************************************************************

def flat_map_512_overlapped_patches_from_harvard_dataset(_img):
    # full-view 8 square patches
    flat_map_operation_list = [
        tf.image.resize(images=tf.image.crop_to_bounding_box(_img, 0, 0, 1040, 1040),
                        size=[512, 512], method=ResizeMethod.BILINEAR),
        tf.image.resize(images=tf.image.crop_to_bounding_box(_img, 0, 352, 1040, 1040),
                        size=[512, 512], method=ResizeMethod.BILINEAR)
    ]
    third_width = 464
    for i in range(0, 1040, 1040 - 512):
        for j in range(0, 1392, third_width):
            flat_map_operation_list.append(safe_crop_to_bounding_box(_img, i, j, 512, 512))
    Logger.i("Harvard cropped patches: %d." % len(flat_map_operation_list))
    return tf.data.Dataset.from_tensor_slices(flat_map_operation_list)


def load_harvard_full_mat(dataset_dir, verbose=True):
    import scipy.io as sio

    def harvard_mat_generator(_mat_list):
        for _mat_path in _mat_list:
            hyper = sio.loadmat(_mat_path)["ref"]
            if verbose:
                tf.print("Decoding MAT: max_val=", tf.reduce_max(hyper), "min_val", tf.reduce_min(hyper),
                         "<shape=", hyper.shape, ">@", _mat_path)
            yield hyper

    calibration = [7.5822230e-02, 1.1085362e-01, 1.5499444e-01, 1.9669126e-01, 2.5034297e-01, 2.7651581e-01,
                   3.1691581e-01, 3.7851196e-01, 4.2183340e-01, 4.5967713e-01, 5.1717121e-01, 5.5704796e-01,
                   5.8606052e-01, 6.6835907e-01, 6.7580378e-01, 6.8838983e-01, 7.0713197e-01, 7.6301691e-01,
                   7.7116192e-01, 7.7402294e-01, 8.0076054e-01, 8.2445726e-01, 9.2331913e-01, 9.2700961e-01,
                   9.2251713e-01, 9.1098958e-01, 9.0504569e-01, 9.0276792e-01, 1.0000000e+00, 9.5739246e-01,
                   8.7174851e-01]
    calibration = tf.reshape(calibration, shape=(1, 1, 31))

    mat_list = tf.io.matching_files(str(dataset_dir) + "/*.mat")

    if verbose:
        Logger.i("Dataset loaded from: ", dataset_dir)
        Logger.i("\n MAT List \n", mat_list)

    mat_dataset = tf.data.Dataset \
        .from_generator(harvard_mat_generator, args=[mat_list],
                        output_types=tf.float32, output_shapes=(1040, 1392, 31)).map(lambda _x: _x / calibration)
    return mat_dataset


def load_harvard_full_mat_512(dataset_dir, cache_name, verbose=True):
    return load_harvard_full_mat(dataset_dir, verbose=verbose) \
        .flat_map(flat_map_512_overlapped_patches_from_harvard_dataset) \
        .cache(cache_name)


# ************************************************************************
# *******************************  KAIST  *********************************
# ************************************************************************
def load_kaist_full_mat(dataset_dir, verbose=True):
    import scipy.io as sio

    def kaist_mat_generator(_mat_list):
        for _mat_path in _mat_list:
            hyper = sio.loadmat(_mat_path)["HSI"]
            tf.print("Decoding MAT: max_val=", tf.reduce_max(hyper), "min_val", tf.reduce_min(hyper),
                     "<shape=", hyper.shape, ">@", _mat_path)
            yield hyper

    mat_list = tf.io.matching_files(str(dataset_dir) + "/*.mat")

    if verbose:
        Logger.i("Dataset loaded from: ", dataset_dir)
        Logger.i("\n ===== MAT List \n", mat_list)

    mat_dataset = tf.data.Dataset \
        .from_generator(kaist_mat_generator, args=[mat_list],
                        output_types=tf.float32, output_shapes=(2704, 3376, 31))

    return mat_dataset


def flat_map_512_patches_from_kaist_dataset(_img):
    flat_map_operation_list = [tf.image.resize(images=_img, size=[512, 512], method=ResizeMethod.BILINEAR),
                               tf.image.resize(images=safe_crop_to_bounding_box(_img, 0, 0, 1024, 1024),
                                               size=[512, 512], method=ResizeMethod.BILINEAR),
                               tf.image.resize(images=safe_crop_to_bounding_box(_img, 0, 512, 1024, 1024),
                                               size=[512, 512], method=ResizeMethod.BILINEAR),
                               tf.image.resize(images=safe_crop_to_bounding_box(_img, 512, 512, 1024, 1024),
                                               size=[512, 512], method=ResizeMethod.BILINEAR),
                               tf.image.resize(images=safe_crop_to_bounding_box(_img, 0, 512, 1024, 1024),
                                               size=[512, 512], method=ResizeMethod.BILINEAR),
                               tf.image.resize(images=safe_crop_to_bounding_box(_img, 512, 0, 1024, 1024),
                                               size=[512, 512], method=ResizeMethod.BILINEAR),
                               tf.image.resize(images=safe_crop_to_bounding_box(_img, 0, 0, 2048, 2048),
                                               size=[512, 512], method=ResizeMethod.BILINEAR),
                               tf.image.resize(images=safe_crop_to_bounding_box(_img, 0, 512, 2048, 2048),
                                               size=[512, 512], method=ResizeMethod.BILINEAR)]
    # for i in range(5):
    #     for j in range(6):
    #         flat_map_operation_list.append(safe_crop_to_bounding_box(_img, 512 * i, 512 * j, 512, 512))
    tf.print("Divided into patches:", len(flat_map_operation_list))
    print("Divided into patches:", len(flat_map_operation_list))
    return tf.data.Dataset.from_tensor_slices(flat_map_operation_list)


def load_kaist_full_mat_512(dataset_dir, cache_name, verbose=True):
    return load_kaist_full_mat(dataset_dir, verbose=verbose).cache(cache_name) \
        .flat_map(flat_map_512_patches_from_kaist_dataset)


# ************************************************************************
# *******************************  MIXED  *********************************
# ************************************************************************
def load_icvl_mat_with_cave_512(dataset_dir, cache_name):
    cave_cache_name = "HS-CAVE-16bit-"
    if "train" in cache_name:
        cave_cache_name = cave_cache_name + "train"
    elif "validation" in cache_name:
        cave_cache_name = cave_cache_name + "validation"
    cave_ds = load_cave_datasets(dataset_dir=DATASET_PATH["CAVE-16bit"], cache_name=cave_cache_name)
    icvl_ds = load_icvl_full_mat_512(dataset_dir, cache_name)
    return icvl_ds.concatenate(cave_ds)


def load_icvl_mat_with_cave_512_shuffled(dataset_dir, cache_name):
    return load_icvl_mat_with_cave_512(dataset_dir, cache_name).shuffle(buffer_size=4)


def load_identical_icvl_harvard_hs_pairs_1024(dataset_dir, cache_name="cache-1024"):
    icvl = []
    harvard = []
    for root, dirs, files in os.walk(dataset_dir):
        for _dir in dirs:
            hyper_png_path_list = tf.io.matching_files(os.path.join(root, _dir, '*.png'))
            png_cnt = 0
            for _ in hyper_png_path_list:
                png_cnt += 1
            if png_cnt != 0 and ("ICVL" in root or "Harvard" in root):
                print("[Lazy Load] Loading ", png_cnt, " PNGs from <", root, "/", _dir, ">")
                if "Harvard" in root:
                    print("\t Harvard will be loaded as 1024*1024 patch;")
                    harvard.append(hyper_png_path_list)
                elif "ICVL" in root:
                    print("\t ICVL will be loaded 1024*1024 patches;")
                    icvl.append(hyper_png_path_list)
            else:
                print("[Lazy Load] Skipped:", root, _dir)
    icvl_ds = tf.data.Dataset.from_tensor_slices(icvl) \
        .map(decode_hyper_image_from_png_list).cache(cache_name + "-icvl")
    harvard_ds = tf.data.Dataset.from_tensor_slices(harvard) \
        .map(decode_hyper_image_from_png_list).cache(cache_name + "-harvard")
    ds = icvl_ds.concatenate(harvard_ds)

    lighten_ds = ds.map(lighten_image)
    final_ds = ds.concatenate(lighten_ds).map(crop_1024)
    return final_ds

