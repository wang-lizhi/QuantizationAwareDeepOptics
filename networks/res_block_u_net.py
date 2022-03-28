import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, UpSampling2D, Add, BatchNormalization, Input, \
    Activation, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model


def _channel_attention_block(x, kernel_size=3, name="attention_blk"):
    attention_conv = Conv1D(filters=1, kernel_size=kernel_size, strides=1,
                            use_bias=False, padding="same", name="%s_attn_conv" % name)
    avg_pool = attention_conv(
        tf.expand_dims(
            GlobalAveragePooling2D(name="%s_attn_avg_pool" % name)(x),
            axis=-1
        )
    )  # shape=(b, c, 1)
    max_pool = attention_conv(
        tf.expand_dims(
            GlobalMaxPooling2D(name="%s_attn_max_pool" % name)(x),
            axis=-1
        )
    )  # shape=(b, c, 1)
    attention_feature = Activation('sigmoid', name="%s_attn_sigmoid" % name)(avg_pool + max_pool)  # shape=(b, c, 1)
    attention_feature = tf.expand_dims(attention_feature, axis=-1)  # shape=(b, c, 1, 1)
    attention_feature = tf.transpose(attention_feature, perm=[0, 2, 3, 1])  # shape=(b, 1, 1, c)
    return attention_feature * x


def get_res_block_u_net(filter_root, depth, output_channel=31, input_size=(512, 512, 3), activation='elu',
                        batch_norm=True, batch_norm_after_activation=False, final_activation='sigmoid', net_num=1,
                        extra_upsampling=False, remove_first_long_connection=False, channel_attention=False,
                        kernel_initializer='glorot_uniform', final_kernel_initializer='glorot_uniform'):
    """
    Our implementation of Res-UNet

    Args:
        filter_root (int): The filter number in first layer.
        depth (int): Depth of UNet. i.e. the count of down/upsampling layers. the `filter_root` and image height and width must be divisible by `2^depth`.
        output_channel (int, optional): Final output channel number.
        input_size (tuple, optional): Input image resolution. Default is (512, 512, 3).
        activation (str, optional): Activation used in layers. Default is 'elu'.
        batch_norm (bool, optional): Whether to add BatchNormalization in each layer. Default is True.
        batch_norm_after_activation: The location of BatchNormalization. Default is before the activation (False).
        final_activation (str, optional): Activation of the last layer. Default is 'sigmoid'；
        net_num (int, optional)：Sub-network count. Default is 1 (only one network).
        extra_upsampling: Whether to add an extra upsampling. This operation can make the output image double the size of the input. Default is False.
        remove_first_long_connection: Whether to discard the long residual connection of the outermost layer of ResUNet. Default is False.
        channel_attention: Whether to use channel attention mechanism on long residual connections. Default is False.
        kernel_initializer: Kernel initializer of convolution layers. Default is 'glorot_uniform'.
        final_kernel_initializer: Kernel initializer of the last layer. Default is 'glorot_uniform'.
    Returns:
        The Res-UNet Keras model.
    """
    assert net_num >= 1, "There should be at least one network."
    print("Network Args:")

    print("activation=", activation)
    print("final_activation=", final_activation)
    print("kernel_initializer=", kernel_initializer)
    print("final_kernel_initializer=", final_kernel_initializer)

    inputs = Input(input_size)
    x = inputs

    # Dictionary for long connections
    long_connection_store = {}

    if net_num == 1 and extra_upsampling:
        x = UpSampling2D(name="first_bilinear_upsampling", interpolation="bilinear")(x)

    for cur_net_num in range(net_num):
        # Downsampling layer (via maxpooling)
        for i in range(depth):
            out_channel = 2 ** i * filter_root
            # Residual/Skip connection
            inner_block_res = Conv2D(out_channel, kernel_size=1, padding='same', use_bias=False,
                                     kernel_initializer=kernel_initializer,
                                     name="net{}_identity{}_1".format(cur_net_num, i))(x)
            # First Conv2D Block with Conv2D, BN and activation
            conv1 = Conv2D(out_channel, kernel_size=7 if i == 0 else 3, padding='same',
                           kernel_initializer=kernel_initializer,
                           name="net{}_conv{}_1".format(cur_net_num, i))(x)
            if batch_norm and not batch_norm_after_activation:
                conv1 = BatchNormalization(name="net{}_bn{}_1".format(cur_net_num, i))(conv1)
            act1 = Activation(activation, name="net{}_act{}_1".format(cur_net_num, i))(conv1)
            if batch_norm and batch_norm_after_activation:
                act1 = BatchNormalization(name="net{}_bn{}_1".format(cur_net_num, i))(act1)
            # Second Conv2D block with Conv2D and BN only
            conv2 = Conv2D(out_channel, kernel_size=3, padding='same',
                           kernel_initializer=kernel_initializer,
                           name="net{}_conv{}_2".format(cur_net_num, i))(act1)
            if batch_norm and not batch_norm_after_activation:
                conv2 = BatchNormalization(name="net{}_bn{}_2".format(cur_net_num, i))(conv2)
            inner_block_res_added = Add(name="net{}_add{}_1".format(cur_net_num, i))([inner_block_res, conv2])
            act2 = Activation(activation, name="net{}_act{}_2".format(cur_net_num, i))(inner_block_res_added)
            if batch_norm and batch_norm_after_activation:
                act2 = BatchNormalization(name="net{}_bn{}_2".format(cur_net_num, i))(act2)
            # Max pooling
            if i < depth - 1:
                if remove_first_long_connection:
                    if i > 0:
                        long_connection_store[str(i)] = act2
                else:
                    long_connection_store[str(i)] = act2
                x = MaxPooling2D(padding='same', name="net{}_maxpooling{}_1".format(cur_net_num, i))(act2)
            else:
                x = act2

        for i in range(depth - 2, -1, -1):
            # Upsampling layers
            out_channel = 2 ** i * filter_root

            up1 = UpSampling2D(name="net{}_up_sampling{}_1".format(cur_net_num, i))(x)
            up_conv1 = Conv2D(out_channel, kernel_size=1, activation=activation, padding='same',
                              kernel_initializer=kernel_initializer,
                              name="net{}_up_conv{}_1".format(cur_net_num, i))(up1)

            if remove_first_long_connection and i == 0:
                # Dispose the first long connection if `remove_first_long_connection` is True.
                up_long_connection_concat = up_conv1
            else:
                long_connection = long_connection_store[str(i)]
                if channel_attention and i < depth-3:
                    # Add channel attention on the long connection if `channel_attention` is True.
                    attention_long_connection = _channel_attention_block(long_connection,
                                                                         name="net{}_long_conn{}".format(cur_net_num,
                                                                                                         i))
                    long_connection = Add(name="net{}_long_conn{}_attn_res_add".format(cur_net_num, i))(
                        [long_connection, attention_long_connection])
                #  Concatenate the long connection
                up_long_connection_concat = Concatenate(axis=-1, name="net{}_up_concat{}_1".format(cur_net_num, i))(
                    [up_conv1, long_connection])

            #  Convolutions
            up_conv2 = Conv2D(out_channel, kernel_size=3, padding='same', kernel_initializer=kernel_initializer,
                              name="net{}_up_conv{}_2".format(cur_net_num, i))(up_long_connection_concat)
            if batch_norm and not batch_norm_after_activation:
                up_conv2 = BatchNormalization(name="net{}_up_bn{}_2".format(cur_net_num, i))(up_conv2)

            up_act1 = Activation(activation, name="net{}_up_act{}_2".format(cur_net_num, i))(up_conv2)

            if batch_norm and batch_norm_after_activation:
                up_act1 = BatchNormalization(name="net{}_up_bn{}_2".format(cur_net_num, i))(up_act1)

            up_conv3 = Conv2D(out_channel, kernel_size=3, padding='same', kernel_initializer=kernel_initializer,
                              name="net{}_up_conv{}_3".format(cur_net_num, i))(up_act1)

            if batch_norm and not batch_norm_after_activation:
                up_conv3 = BatchNormalization(name="net{}_up_bn{}_3".format(cur_net_num, i))(up_conv3)

            # Residual/Skip connection
            inner_block_res = Conv2D(out_channel, kernel_size=1, padding='same', use_bias=False,
                                     kernel_initializer=kernel_initializer,
                                     name="net{}_up_identity{}_1".format(cur_net_num, i))(up_long_connection_concat)

            inner_block_res_added = Add(name="net{}_up_concat{}_res".format(cur_net_num, i))(
                [inner_block_res, up_conv3])

            x = Activation(activation, name="net{}_up_act{}_res".format(cur_net_num, i))(inner_block_res_added)
            if batch_norm and batch_norm_after_activation:
                x = BatchNormalization(name="net{}_up_bn{}_3".format(cur_net_num, i))(x)

        if net_num > 1 and cur_net_num == 0 and extra_upsampling:
            x = UpSampling2D(name="middle_res_up_sampling", interpolation="bilinear")(x)

    # Final convolution
    output = Conv2D(output_channel, 1, padding='same', activation=final_activation,
                    kernel_initializer=final_kernel_initializer, name='output')(x)
    model = Model(inputs, outputs=output, name='res-block-u-net')
    # Regularization (L2)
    for layer in model.layers:
        if "kernel_regularizer" in layer.__dict__:
            layer.kernel_regularizer = tf.keras.regularizers.l2(l2=0.0001)
    return model
