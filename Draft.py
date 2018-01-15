import tensorflow as tf
with tf.variable_scope("scope"):
    sess = tf.Session()
    weight = tf.get_variable("weight", [1, 1, 256, 256], initializer=tf.contrib.layers.variance_scaling_initializer())
    in_channels = int(weight.get_shape()[-2])
    d_in = in_channels // 4
    d_out = int(weight.get_shape()[-1]) // 4
    zeros = tf.zeros([d_out])
    init = tf.global_variables_initializer()
    sess.run(init)
    weight_s = tf.abs(tf.squeeze(weight))
    k = in_channels
    # Sort and Drop
    for group in range(4):
        wi = weight_s[:, group * d_out:(group + 1) * d_out]
        # take corresponding delta index
        _, index = tf.nn.top_k(tf.reduce_sum(wi, axis=1), k=k, sorted=True)
        d = sess.run(index)
        for _in in range(d_in):
            # Assume only apply to 1x1 conv to speed up
            print("group:%d,_in:%d" % (group, _in))
            sess.run(tf.assign(weight[0, 0, d[-(_in + 1)], group * d_out:(group + 1) * d_out], zeros))
    print(sess.run(weight))




