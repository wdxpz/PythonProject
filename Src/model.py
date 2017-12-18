from Src.network import conv2d
import tensorflow as tf
import math


def model_input():
    inputs = tf.placeholder(tf.float32, shape=[1, None, None, 1], name='input_image')
    gt = tf.placeholder(tf.float32, shape=[1, None, None, 1], name='ground_true')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    is_training = tf.placeholder(tf.bool, name='is_training')

    return inputs, gt, learning_rate, is_training


def model_MCNN(inputs, bn, dropout, is_train):
    # branch1
    branch1 = conv2d(inputs, 16, 9, 1, bn=bn, is_train=is_train, relu=True, dropout=dropout, max_pool=True)
    branch1 = conv2d(branch1, 32, 7, 1, bn=bn, is_train=is_train, relu=True, dropout=dropout, max_pool=True)
    branch1 = conv2d(branch1, 16, 7, 1, bn=bn, is_train=is_train, relu=True, dropout=dropout, max_pool=False)
    branch1 = conv2d(branch1, 8, 7, 1, bn=bn, is_train=is_train, relu=True, dropout=dropout, max_pool=False)

    # branch2
    branch2 = conv2d(inputs, 20, 7, 1, bn=bn, is_train=is_train, relu=True, dropout=dropout, max_pool=True)
    branch2 = conv2d(branch2, 40, 5, 1, bn=bn, is_train=is_train, relu=True, dropout=dropout, max_pool=True)
    branch2 = conv2d(branch2, 20, 3, 1, bn=bn, is_train=is_train, relu=True, dropout=dropout, max_pool=False)
    branch2 = conv2d(branch2, 10, 3, 1, bn=bn, is_train=is_train, relu=True, dropout=dropout, max_pool=False)

    # branch3
    branch3 = conv2d(inputs, 24, 5, 1, bn=bn, is_train=is_train, relu=True, dropout=dropout, max_pool=True)
    branch3 = conv2d(branch3, 48, 3, 1, bn=bn, is_train=is_train, relu=True, dropout=dropout, max_pool=True)
    branch3 = conv2d(branch3, 24, 3, 1, bn=bn, is_train=is_train, relu=True, dropout=dropout, max_pool=False)
    branch3 = conv2d(branch3, 12, 3, 1, bn=bn, is_train=is_train, relu=True, dropout=dropout, max_pool=False)

    fuse = tf.concat([branch1, branch2, branch3], 3)

    # final output
    digits = conv2d(fuse, 1, 1, 1, bn=False, is_train=is_train, relu=False, dropout=False, max_pool=False)

    return digits


def model_loss(ground_true, density_map):
    # gt_shape = ground_true.get_shape().as_list()
    # dt_shape = density_map.get_shape().as_list()
    resize_shape = tf.shape(tf.squeeze(density_map, [0, 3]))
    resize_gt = tf.image.resize_images(ground_true, resize_shape) \
                * (tf.to_float(tf.size(ground_true)) / tf.to_float(tf.size(density_map)))
    # !!!!!!!!!!!!!!validation
    gt_count = tf.reduce_sum(ground_true)
    dt_count = tf.reduce_sum(density_map)
    re_count = tf.reduce_sum(resize_gt)

    # mse = tf.losses.mean_squared_error(resize_gt, density_map)
    # loss = tf.reduce_mean(mse)
    mae = tf.abs(dt_count-gt_count)
    # mse = tf.sqrt(tf.reduce_mean((resize_gt - density_map)*(resize_gt - density_map)))
    return mae, gt_count, dt_count, re_count


def model_crowd_count(ground_true, density_map):
    return tf.reduce_sum(ground_true), tf.reduce_sum(density_map)


def model_opt(loss, learning_rate, beta1, bn=False):
    """
        Get optimization operations
        :param oss: Discriminator loss Tensor
        :param learning_rate: Learning Rate Placeholder
        :param beta1: The exponential decay rate for the 1st moment in the optimizer
        :return: training operation
        """
    if bn:
        t_vars = tf.trainable_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss, var_list=t_vars)
    else:
        train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss)

    return train_opt
