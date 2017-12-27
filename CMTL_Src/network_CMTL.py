import tensorflow as tf
import numpy as np


def conv2d(inputs, out_channels, kernel_size, stride, relu=True, dropout=False,
           max_pool=True, alpha=0.02, keep_prob=0.8, max_pool_size=2):
    kernel_initializer = tf.contrib.layers.xavier_initializer()
    conv = tf.layers.conv2d(inputs, out_channels, kernel_size, stride, padding='same', activation=None,
                            kernel_initializer= kernel_initializer)
    if relu == 'leaky_relu':
        conv = tf.maximum(alpha * conv, conv)
    elif relu == 'prelu':
        conv = prelu(conv)
    elif relu == 'relu':
        conv = tf.nn.relu(conv)

    if dropout:
        conv = tf.nn.dropout(conv, keep_prob=keep_prob)

    if max_pool:
        conv = tf.layers.max_pooling2d(conv, max_pool_size, max_pool_size, padding='same')

    return conv


def conv2d_trans(inputs, out_channels, kernel_size, stride, relu=True, dropout=False, alpha=0.02,
                 keep_prob=0.8):
    conv = tf.layers.conv2d_transpose(inputs, out_channels, kernel_size, strides=stride, padding='same',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
    if relu == 'leaky_relu':
        conv = tf.maximum(alpha * conv, conv)
    elif relu == 'prelu':
        conv = prelu(conv)
    elif relu == 'relu':
        conv = tf.nn.relu(conv)

    if dropout:
        conv = tf.nn.dropout(conv, keep_prob=keep_prob)

    return conv


def fc(inputs, out_features, relu=True, alpha=0.02, dropout=False, keep_prob=0.8):
    digits = tf.contrib.layers.fully_connected(inputs, out_features, activation_fn=None)
    if relu:
        # digits = tf.nn.relu(digits)
        digits = tf.maximum(alpha * digits, digits)
    if dropout:
        digits = tf.nn.dropout(digits, keep_prob=keep_prob)
    return digits


def spp_layer2(input_, levels=[2, 1], name='SPP_layer'):
    '''Multiple Level SPP layer.
       Works for levels=[1, 2, 3, 6].'''
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        pool_outputs = []
        for l in levels:
            pool = tf.nn.max_pool(input_, ksize=[1, np.ceil(shape[1] * 1. / l).astype(np.int32),
                                                 np.ceil(shape[2] * 1. / l).astype(np.int32), 1],
                                  strides=[1, np.floor(shape[1] * 1. / l + 1).astype(np.int32),
                                           np.floor(shape[2] * 1. / l + 1), 1],
                                  padding='SAME')
            print("Pool Level {:}: shape {:}".format(l, pool.get_shape().as_list()))
            pool_outputs.append(tf.reshape(pool, [shape[0], -1]))
        spp_pool = tf.concat(pool_outputs, 1)
    return spp_pool


def max_pool_2d_nxn_regions(inputs, output_size: int, mode: str):
    """
    Performs a pooling operation that results in a fixed size:
    output_size x output_size.

    Used by spatial_pyramid_pool. Refer to appendix A in [1].

    Args:
        inputs: A 4D Tensor (B, H, W, C)
        output_size: The output size of the pooling operation.
        mode: The pooling mode {max, avg}

    Returns:
        A list of tensors, for each output bin.
        The list contains output_size * output_size elements, where
        each elment is a Tensor (N, C).

    References:
        [1] He, Kaiming et al (2015):
            Spatial Pyramid Pooling in Deep Convolutional Networks
            for Visual Recognition.
            https://arxiv.org/pdf/1406.4729.pdf.

    Ported from: https://github.com/luizgh/Lasagne/commit/c01e3d922a5712ca4c54617a15a794c23746ac8c
    """
    inputs_shape = tf.shape(inputs)
    h = tf.cast(tf.gather(inputs_shape, 1), tf.int32)
    w = tf.cast(tf.gather(inputs_shape, 2), tf.int32)

    if mode == 'max':
        pooling_op = tf.reduce_max
    elif mode == 'avg':
        pooling_op = tf.reduce_mean
    else:
        msg = "Mode must be either 'max' or 'avg'. Got '{0}'"
        raise ValueError(msg.format(mode))

    result = []
    n = output_size
    for row in range(output_size):
        for col in range(output_size):
            # start_h = floor(row / n * h)
            start_h = tf.cast(tf.floor(tf.multiply(tf.divide(row, n), tf.cast(h, tf.float32))), tf.int32)
            # end_h = ceil((row + 1) / n * h)
            end_h = tf.cast(tf.ceil(tf.multiply(tf.divide((row + 1), n), tf.cast(h, tf.float32))), tf.int32)
            # start_w = floor(col / n * w)
            start_w = tf.cast(tf.floor(tf.multiply(tf.divide(col, n), tf.cast(w, tf.float32))), tf.int32)
            # end_w = ceil((col + 1) / n * w)
            end_w = tf.cast(tf.ceil(tf.multiply(tf.divide((col + 1), n), tf.cast(w, tf.float32))), tf.int32)
            pooling_region = inputs[:, start_h:end_h, start_w:end_w, :]
            pool_result = pooling_op(pooling_region, axis=(1, 2))
            result.append(pool_result)
    return result


def spatial_pyramid_pool(inputs, dimensions=[2, 1], mode='max', implementation='kaiming'):
    """
    Performs spatial pyramid pooling (SPP) over the input.
    It will turn a 2D input of arbitrary size into an output of fixed
    dimenson.
    Hence, the convlutional part of a DNN can be connected to a dense part
    with a fixed number of nodes even if the dimensions of the input
    image are unknown.

    The pooling is performed over :math:`l` pooling levels.
    Each pooling level :math:`i` will create :math:`M_i` output features.
    :math:`M_i` is given by :math:`n_i * n_i`, with :math:`n_i` as the number
    of pooling operations per dimension level :math:`i`.

    The length of the parameter dimensions is the level of the spatial pyramid.

    Args:
        inputs: A 4D Tensor (B, H, W, C).
        dimensions: The list of :math:`n_i`'s that define the output dimension
        of each pooling level :math:`i`. The length of dimensions is the level of
        the spatial pyramid.
        mode: Pooling mode 'max' or 'avg'.
        implementation: The implementation to use, either 'kaiming' or 'fast'.
        kamming is the original implementation from the paper, and supports variable
        sizes of input vectors, which fast does not support.

    Returns:
        A fixed length vector representing the inputs.

    Notes:
        SPP should be inserted between the convolutional part of a DNN and it's
        dense part. Convolutions can be used for arbitrary input dimensions, but
        the size of their output will depend on their input dimensions.
        Connecting the output of the convolutional to the dense part then
        usually demands us to fix the dimensons of the network's input.
        The spatial pyramid pooling layer, however, allows us to leave
        the network input dimensions arbitrary.
        The advantage over a global pooling layer is the added robustness
        against object deformations due to the pooling on different scales.

    References:
        [1] He, Kaiming et al (2015):
            Spatial Pyramid Pooling in Deep Convolutional Networks
            for Visual Recognition.
            https://arxiv.org/pdf/1406.4729.pdf.

    Ported from: https://github.com/luizgh/Lasagne/commit/c01e3d922a5712ca4c54617a15a794c23746ac8c
    """
    pool_list = []
    if implementation == 'kaiming':
        for pool_dim in dimensions:
            pool_list += max_pool_2d_nxn_regions(inputs, pool_dim, mode)
    else:
        shape = inputs.get_shape().as_list()
        for d in dimensions:
            h = shape[1]
            w = shape[2]
            ph = np.ceil(h * 1.0 / d).astype(np.int32)
            pw = np.ceil(w * 1.0 / d).astype(np.int32)
            sh = np.floor(h * 1.0 / d + 1).astype(np.int32)
            sw = np.floor(w * 1.0 / d + 1).astype(np.int32)
            pool_result = tf.nn.max_pool(inputs,
                                         ksize=[1, ph, pw, 1],
                                         strides=[1, sh, sw, 1],
                                         padding='SAME')
            pool_list.append(tf.reshape(pool_result, [tf.shape(inputs)[0], -1]))
    return tf.concat(pool_list, 1) #tf.concat(1, pool_list)


def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)
