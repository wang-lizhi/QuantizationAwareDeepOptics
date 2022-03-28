import tensorflow as tf


def save_numpy_to_file(file_name, file_content):
    with tf.io.gfile.GFile(file_name, 'wb') as file:
        file.write(file_content)
        file.flush()
