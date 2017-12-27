import tensorflow as tf


def conv2d(inputs, out_channels, kernel_size, stride, relu=True, dropout=False,
           max_pool=True, alpha=0.02, keep_prob=0.8, max_pool_size=2):
    kernel_initializer = tf.contrib.layers.xavier_initializer()
    conv = tf.layers.conv2d(inputs, out_channels, kernel_size, stride, padding='same', activation=None,
                            kernel_initializer= kernel_initializer)
    if relu:
        conv = tf.nn.relu(conv)
        # conv = tf.maximum(alpha * conv, conv)

    if dropout:
        conv = tf.nn.dropout(conv, keep_prob=keep_prob)

    if max_pool:
        conv = tf.layers.max_pooling2d(conv, max_pool_size, max_pool_size, padding='same')

    return conv


def fc(inputs, out_features, bn=False, is_train=True, relu=True, alpha=0.02, dropout=False, keep_prob=0.8):
    digits = tf.layers.fully_connected(inputs, out_features, activation_fn=None)
    if bn:
        digits = tf.layers.batch_normalization(digits, training=is_train)
    if relu:
        # digits = tf.nn.relu(digits)
        digits = tf.maximum(alpha * digits, digits)
    if dropout:
        digits = tf.nn.dropout(digits, keep_prob=keep_prob)
    return digits

