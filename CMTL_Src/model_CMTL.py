from CMTL_Src.network_CMTL import *
import tensorflow as tf
import math


def model_input(batch_size, num_classes):
    # inputs = tf.placeholder(tf.float32, shape=[batch_size, None, None, 1], name='input_image')
    # gt_density = tf.placeholder(tf.float32, shape=[batch_size, None, None, 1], name='gt_density')
    ####only for dataset B
    inputs = tf.placeholder(tf.float32, shape=[batch_size, 384, 512, 1], name='input_image')
    gt_density = tf.placeholder(tf.float32, shape=[batch_size, 384, 512, 1], name='gt_density')
    gt_label = tf.placeholder(tf.float32, shape=[batch_size, num_classes], name='gt_label')
    ce_weights = tf.placeholder(tf.float32, shape=[batch_size], name='gt_label')

    return inputs, gt_density, gt_label, ce_weights


def CMTL(inputs, dropout, num_classes=10):

    #base layer
    base_layer = conv2d(inputs, 16, 9, 1, relu='prelu', dropout=False, max_pool=False)
    base_layer = conv2d(base_layer, 32, 7, 1, relu='prelu', dropout=False, max_pool=False)

    #high-level prior stage1
    hl_prior_1 = conv2d(base_layer, 16, 9, 1, relu='prelu', dropout=False, max_pool=True)
    hl_prior_1 = conv2d(hl_prior_1, 32, 7, 1, relu='prelu', dropout=False, max_pool=True)
    hl_prior_1 = conv2d(hl_prior_1, 16, 7, 1, relu='prelu', dropout=False, max_pool=False)
    hl_prior_1 = conv2d(hl_prior_1, 8, 7, 1, relu='prelu', dropout=False, max_pool=False)

    # high-level prior pyramid max pooling
    # hl_prior_2 = spatial_pyramid_pool(hl_prior_1, dimensions=[16, 16])
    hl_prior_2 = spp_layer2(hl_prior_1, [32])
    # print('tenser after pyramid_pooling: {}'.format(hl_prior_2))
    # to check if the ourput is 4*1024

    #high-level prior stage 2
    hl_prior_fc = fc(hl_prior_2, 512, relu='prelu', dropout=True)
    hl_prior_fc = fc(hl_prior_fc, 256, relu='prelu', dropout=True)
    class_logits = fc(hl_prior_fc, num_classes, relu='none', dropout=False)
    #???????for the classification logits, is it need to user activation?????
    # class_logits = tf.nn.softmax(class_logits)


    #density estimation stage
    den_stage_1 = conv2d(base_layer, 20, 7, 1, relu='prelu', dropout=False, max_pool=True)
    den_stage_1 = conv2d(den_stage_1, 40, 5, 1, relu='prelu', dropout=False, max_pool=True)
    den_stage_1 = conv2d(den_stage_1, 20, 5, 1, relu='prelu', dropout=False, max_pool=False)
    den_stage_1 = conv2d(den_stage_1, 10, 5, 1, relu='prelu', dropout=False, max_pool=False)

    fuse = tf.concat([hl_prior_1, den_stage_1], 3)

    den_stage_2 = conv2d(fuse, 24, 3, 1, relu='prelu', dropout=False, max_pool=False)
    den_stage_2 = conv2d(den_stage_2, 32, 3, 1, relu='prelu', dropout=False, max_pool=False)
    den_stage_2 = conv2d_trans(den_stage_2, 16, 4, 2, relu='prelu', dropout=False)
    den_stage_2 = conv2d_trans(den_stage_2, 8, 4, 2, relu='prelu', dropout=False)

    den = conv2d(den_stage_2, 1, 1, 1, relu='none', dropout=False, max_pool=False)
    #where the original paper used relu, not None nor prelu
    #????????for the output of density map, it is need to user activation????????/

    return class_logits, den


def model_loss(class_lable, class_logits, ground_true, density_map, ce_weight=1.0):
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(class_lable, class_logits)
    cross_entropy = tf.losses.softmax_cross_entropy(class_lable, class_logits, ce_weight)

    mae = tf.reduce_mean(tf.abs(tf.reduce_sum(ground_true, (1, 2)) - tf.reduce_sum(density_map, (1, 2))))
    mse = tf.sqrt(tf.reduce_mean(tf.squared_difference(tf.reduce_sum(ground_true, (1, 2)),
                                                       tf.reduce_sum(density_map, (1, 2)))))
    loss_den = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(ground_true, density_map), (1, 2))))

    loss = loss_den + 0.0001 * cross_entropy

    return loss, cross_entropy, loss_den, mae, mse


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
