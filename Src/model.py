from Src.network import conv2d
import tensorflow as tf


def model_input(batch_size):
    inputs = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_image')
    gt = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='ground_true')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    return inputs, gt, learning_rate



def model_MCNN(inputs, dropout):
    # branch1
    branch1 = conv2d(inputs, 16, 9, 1, relu=True, dropout=dropout, max_pool=True)
    branch1 = conv2d(branch1, 32, 7, 1, relu=True, dropout=dropout, max_pool=True)
    branch1 = conv2d(branch1, 16, 7, 1, relu=True, dropout=dropout, max_pool=False)
    branch1 = conv2d(branch1, 8, 7, 1,  relu=True, dropout=dropout, max_pool=False)

    # branch2
    branch2 = conv2d(inputs, 20, 7, 1, relu=True, dropout=dropout, max_pool=True)
    branch2 = conv2d(branch2, 40, 5, 1, relu=True, dropout=dropout, max_pool=True)
    branch2 = conv2d(branch2, 20, 3, 1, relu=True, dropout=dropout, max_pool=False)
    branch2 = conv2d(branch2, 10, 3, 1, relu=True, dropout=dropout, max_pool=False)

    # branch3
    branch3 = conv2d(inputs, 24, 5, 1, relu=True, dropout=dropout, max_pool=True)
    branch3 = conv2d(branch3, 48, 3, 1, relu=True, dropout=dropout, max_pool=True)
    branch3 = conv2d(branch3, 24, 3, 1, relu=True, dropout=dropout, max_pool=False)
    branch3 = conv2d(branch3, 12, 3, 1, relu=True, dropout=dropout, max_pool=False)

    fuse = tf.concat([branch1, branch2, branch3], 3)

    # final output
    digits = conv2d(fuse, 1, 1, 1, relu=False, dropout=False, max_pool=False)

    return digits


def model_loss(ground_true, density_map):
    '''
    because the ground true desity map has already resize to the same size as the generated density map,
    t is no need to resize the ground true
    resize_shape = tf.shape(tf.squeeze(density_map, [0, 3]))
    resize_gt = tf.image.resize_images(ground_true, resize_shape) \
                * (tf.to_float(tf.size(ground_true)) / tf.to_float(tf.size(density_map)))
    '''

    # !!!!!!!!!!!!!!validation
    # gt_count = tf.reduce_sum(ground_true)
    # dt_count = tf.reduce_sum(density_map)
    mae = tf.reduce_mean(tf.abs(tf.reduce_sum(ground_true, (1, 2)) - tf.reduce_sum(density_map, (1, 2))))
    mse = tf.sqrt(tf.reduce_mean(tf.squared_difference(tf.reduce_sum(ground_true, (1, 2)),
                                                       tf.reduce_sum(density_map, (1, 2)))))
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(ground_true, density_map),(1, 2))))
    return loss, mae, mse


def model_crowd_count(ground_true, density_map):
    return tf.reduce_sum(ground_true), tf.reduce_sum(density_map)


def model_opt(loss, learning_rate):
    """
        Get optimization operations
        :param oss: Discriminator loss Tensor
        :param learning_rate: Learning Rate Placeholder
        :param beta1: The exponential decay rate for the 1st moment in the optimizer
        :return: training operation
        """
    # if bn:
    #     t_vars = tf.trainable_variables()
    #     with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    #         train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss, var_list=t_vars)
    # else:
    #     train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss)

    train_opt = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
    return train_opt
