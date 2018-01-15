import tensorflow as tf
with tf.variable_scope("scope"):
    sess = tf.Session()
    weight = tf.get_variable("weight", [1, 1, 32, 16], initializer=tf.contrib.layers.variance_scaling_initializer())
    weight_s = tf.abs(tf.squeeze(weight))
    weight_s1 = weight_s[:, 8:16]
    _, index = tf.nn.top_k(tf.reduce_sum(weight_s1, axis=1), k=16, sorted=True)
    init = tf.global_variables_initializer()
with tf.variable_scope("scope",reuse=True):
    weight_r = tf.get_variable("weight")


print(sess.run(weight_r))
