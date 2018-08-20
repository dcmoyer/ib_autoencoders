""" Architecture specs for CorEx with Echo Noise
"""
import numpy as np
import tensorflow as tf  # TF 1.0 or greater


# Base specification picks out the right encoder/decoder based on architecture specification
def build_encoder(input_tensor, architecture_args, reuse=False):
    """Build the encoder graph."""
    if architecture_args.get('type', 'fc') == 'fc':
        return build_fc_encoder(input_tensor, architecture_args, reuse=reuse)
    elif architecture_args['type'] == 'conv':
        return build_conv_encoder(input_tensor, architecture_args, reuse=reuse)
    else:
        print('Type: {} not recognized'.format(architecture_args['type']))


def build_decoder(encoder, architecture_args, data_shape, reuse=False):
    """Build the decoder graph."""
    if architecture_args.get('type', 'fc') == 'fc':
        return build_fc_decoder(encoder, architecture_args, data_shape, reuse=reuse)
    elif architecture_args['type'] == 'conv':
        return build_conv_decoder(encoder, architecture_args, data_shape, reuse=reuse)
    else:
        print('Type: {} not recognized'.format(architecture_args['type']))


# FC: FULLY CONNECTED
def build_fc_encoder(input_tensor, architecture_args, reuse=False):
    """Build the encoder graph."""
    in_dim_flat = np.prod(input_tensor.get_shape().as_list()[1:])
    all_layer_sizes = [in_dim_flat] + architecture_args.get('layers', [100, 20])
    activation = architecture_args['activation']

    with tf.variable_scope('encoder', reuse=reuse):
        encoder = tf.reshape(input_tensor, (-1, in_dim_flat), name='flatten_input')
        for ilayer in range(1, len(all_layer_sizes)):
            with tf.variable_scope(str(ilayer - 1)):
                encoder = tf.layers.dense(encoder, units=all_layer_sizes[ilayer],
                                          activation=activation, name='dense')
    return encoder


def build_fc_decoder(noisy_encoder, architecture_args, data_shape, reuse=False):
    """Fully connected decoder."""
    all_layer_sizes = [np.prod(data_shape)] + architecture_args.get('layers', [100, 20])
    activation = architecture_args['activation']

    with tf.variable_scope('decoder', reuse=reuse):
        decoder = noisy_encoder
        for ilayer in range(len(all_layer_sizes) - 2, -1, -1):
            with tf.variable_scope(str(ilayer)):
                if ilayer == 0:
                    # Either the decoder is interpreted as logits or continuous. Either way, omit the nonlinearity.
                    decoder = tf.layers.dense(decoder, units=all_layer_sizes[ilayer], activation=None, name='dense')
                    decoder = tf.reshape(decoder, (-1,) + data_shape, name="reshape_to_input")
                else:
                    decoder = tf.layers.dense(decoder, units=all_layer_sizes[ilayer],
                                              activation=activation, name='dense')
    return decoder


# CONVOLUTIONS
def build_conv_encoder(input_tensor, architecture_args, reuse=False):
    """Build a convolutional encoder graph. Mostly copied from Shuyang's corex autoencoder."""
    activation = architecture_args['activation']
    dim = architecture_args.get('dim', 32)
    # TODO: add batch norm option?
    in_shape = input_tensor.get_shape().as_list()
    if len(in_shape) == 3:
        encoder = tf.reshape(input_tensor, [-1, in_shape[1], in_shape[2], 1])
    else:
        encoder = input_tensor
    with tf.variable_scope('encoder', reuse=reuse):
        encoder = tf.layers.conv2d(encoder, 1 * dim, (5, 5), strides=(2, 2), padding='same', activation=activation)
        print(encoder.get_shape().as_list())
        encoder = tf.layers.conv2d(encoder, 2 * dim, (5, 5), strides=(2, 2), padding='same', activation=activation)
        print(encoder.get_shape().as_list())
        encoder = tf.layers.conv2d(encoder, 4 * dim, (5, 5), strides=(2, 2), padding='same', activation=activation)
        print(encoder.get_shape().as_list())
        encoder = tf.layers.conv2d(encoder, 8 * dim, (5, 5), strides=(2, 2), padding='same', activation=activation)
        print(encoder.get_shape().as_list())
    return encoder


def build_conv_decoder(encoder, architecture_args, data_shape, reuse=False):
    """Fully connected decoder."""
    if len(data_shape) == 3:
        channels = data_shape[2]
    else:
        channels = 1
    activation = architecture_args['activation']
    dim = architecture_args.get('dim', 32)
    with tf.variable_scope('decoder', reuse=reuse):
        decoder = tf.layers.conv2d_transpose(encoder, 4 * dim, (5, 5), strides=(2, 2), padding='same', activation=activation)
        print(decoder.get_shape().as_list())
        decoder = tf.layers.conv2d_transpose(decoder, 2 * dim, (5, 5), strides=(2, 2), padding='same', activation=activation)
        print(decoder.get_shape().as_list())
        decoder = tf.layers.conv2d_transpose(decoder, 1 * dim, (5, 5), strides=(2, 2), padding='same', activation=activation)
        print(decoder.get_shape().as_list())
        decoder = tf.layers.conv2d_transpose(decoder, channels, (5, 5), strides=(2, 2), padding='same', activation=None)  # No activation on output decoder
        print(decoder.get_shape().as_list())
        h, w = decoder.get_shape().as_list()[1:3]
        print(h,w, data_shape)
        if [h, w] != data_shape[:2]:
            extra1 = h - data_shape[0]
            extra2 = w - data_shape[1]
            decoder = decoder[:, int(extra1/2):(h - int(extra1/2)), int(extra2/2):(w - int(extra2/2)),:]
        print(decoder.get_shape().as_list())
        decoder = tf.reshape(decoder, (-1,) + data_shape, name="reshape_to_input")
        print(decoder.get_shape().as_list())
    return decoder
