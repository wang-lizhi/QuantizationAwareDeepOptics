import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod


def decode_jpg(file_path):
    image_file = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(contents=image_file, channels=0)
    tf.print("Decoding JPG: <shape=", image.shape, ">@", file_path)
    image = tf.cast(image, tf.float32)  # Shape [height, width, 1]
    image /= 255.0
    return image


def decode_rgb_8bit_png(file_path):
    png_image = tf.image.decode_png(contents=tf.io.read_file(file_path), channels=3, dtype=tf.uint8)
    tf.print("Decoding 8-bit RGB PNG: max_val=", tf.reduce_max(png_image), "min_val", tf.reduce_min(png_image),
             "<shape=", png_image.shape, ">@", file_path)
    png_image = tf.cast(png_image, tf.float32)
    png_image /= 255
    return png_image


def decode_grayscale_16bit_png(file_path):
    image = tf.io.read_file(file_path)
    png_image = tf.image.decode_png(contents=image, channels=1, dtype=tf.uint16)
    tf.print("Decoding 16-bit PNG: max_val=", tf.reduce_max(png_image), "min_val", tf.reduce_min(png_image),
             "<shape=", png_image.shape, ">@", file_path)
    png_image = tf.cast(png_image, tf.float32)
    png_image /= 65535
    return png_image


def decode_hyper_image_from_png_list(image_list):
    hyper_image_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    file_index = 0
    for png_file in image_list:
        single_grayscale_channel = decode_grayscale_16bit_png(png_file)
        hyper_image_array = hyper_image_array.write(file_index, single_grayscale_channel)
        file_index += 1
    hyper_image = hyper_image_array.stack()
    hyper_image = tf.squeeze(hyper_image, axis=-1)
    hyper_image = tf.transpose(hyper_image, perm=[1, 2, 0])
    return hyper_image


def safe_crop_to_bounding_box(image, offset_h, offset_w, target_h, target_w):
    image_shape = tf.shape(image)
    height = image_shape[0]
    width = image_shape[1]
    if offset_w + target_w > width:
        offset_w = width - target_w
        tf.print("[Safe Cropping Warning] offset_w + target_w > width. Cropping inner area...")
        tf.print(offset_w, target_w)
    if offset_h + target_h > height:
        offset_h = height - target_h
        tf.print("[Safe Cropping Warning] offset_h + target_h > height，Cropping inner area...")
        tf.print(offset_h, target_h)
    return tf.image.crop_to_bounding_box(image, offset_h, offset_w, target_w, target_h)


def flat_map_256_patches_from_dataset(_img):
    flat_map_operation_list = [tf.image.resize(images=_img, size=[256, 256], method=ResizeMethod.NEAREST_NEIGHBOR)]
    for i in range(0, 1024, 256):
        for j in range(0, 1024, 256):
            flat_map_operation_list.append(safe_crop_to_bounding_box(_img, i, j, 256, 256))
    return tf.data.Dataset.from_tensor_slices(flat_map_operation_list)


def flat_map_512_patches_from_dataset(_img):
    flat_map_operation_list = [tf.image.resize(images=_img, size=[512, 512], method=ResizeMethod.NEAREST_NEIGHBOR)]
    for i in range(0, 1024, 512):
        for j in range(0, 1024, 512):
            flat_map_operation_list.append(safe_crop_to_bounding_box(_img, i, j, 512, 512))
    return tf.data.Dataset.from_tensor_slices(flat_map_operation_list)


def lighten_image(image):
    _max = tf.reduce_max(image)
    if _max <= 0:
        return image
    return image / _max


def crop_1024(image):
    image_shape = tf.shape(image)
    height = image_shape[0]
    width = image_shape[1]
    if height < 1024 or width < 1024:
        padding_height = 1024 - height
        padding_width = 1024 - width
        if padding_width < 0:
            padding_width = 0
        if padding_height < 0:
            padding_height = 0
        tf.print("[Crop 1024] Image height or width is less than 1024, perform padding：p_h=", padding_height,
                 "; p_w=", padding_width)
        image = tf.pad(image, paddings=[[0, padding_height], [0, padding_width], [0, 0]], mode="SYMMETRIC")
    cropped = tf.image.crop_to_bounding_box(image, 0, 0, 1024, 1024)
    return cropped


def flat_map_image_self_and_transpose(_img):
    return tf.data.Dataset.from_tensor_slices([
        _img,
        tf.transpose(_img, perm=[1, 0, 2])
    ])


def crop_cave_to_5_patches(_img):
    return tf.data.Dataset.from_tensor_slices([
        tf.image.resize(images=_img, size=[256, 256], method=ResizeMethod.NEAREST_NEIGHBOR),
        tf.image.crop_to_bounding_box(_img, 0, 0, 256, 256),
        tf.image.crop_to_bounding_box(_img, 256, 0, 256, 256),
        tf.image.crop_to_bounding_box(_img, 0, 256, 256, 256),
        tf.image.crop_to_bounding_box(_img, 256, 256, 256, 256)
    ])
