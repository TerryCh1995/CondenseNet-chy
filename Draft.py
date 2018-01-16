import tensorflow as tf
import numpy as np
with tf.variable_scope("scope"):
    # sess = tf.Session()
    # weight = tf.get_variable("weight", [1, 1, 256, 256], initializer=tf.contrib.layers.variance_scaling_initializer())
    # mask = tf.get_variable("mask", [1, 1, 256, 256], initializer=tf.constant_initializer(1))
    # in_channels = int(weight.get_shape()[-2])
    # d_in = in_channels // 4
    # d_out = int(weight.get_shape()[-1]) // 4
    # zeros = np.zeros([d_out])
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # weight_array = sess.run(weight)
    # weight_array = np.abs(np.squeeze(weight_array))
    #
    # k = in_channels
    # # Sort and Drop
    # for group in range(4):
    #     wi = weight_array[:, group * d_out:(group + 1) * d_out]
    #
    #     # take corresponding delta index
    #     index = np.argsort(-(wi.sum(1)))
    #
    #     for _in in range(d_in):
    #         sess.run(tf.assign(mask[0, 0, index[k - 1 - _in], group * d_out:(group + 1) * d_out], zeros))
    #         print(group,_in)
    sess = tf.Session()
    one = tf.ones([2, 3, 4])
    print(sess.run(tf.reduce_sum(one)))



