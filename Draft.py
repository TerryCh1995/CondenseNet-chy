import tensorflow as tf
sess = tf.Session()
mask = tf.get_variable("mask", shape=[4], initializer=tf.constant_initializer(0))
sess.run(tf.global_variables_initializer())
weight = tf.get_variable("weight", [1, 1, 32, 16], initializer=tf.contrib.layers.variance_scaling_initializer())
weight_s = tf.abs(tf.squeeze(weight))
weight_s1 = weight_s[:, 8:16]
_, index = tf.nn.top_k(tf.reduce_sum(weight_s1, axis=1), k=16, sorted=True)
init = tf.global_variables_initializer()
sess.run(init)
for i in range(4):
    d = sess.run(index[-(i+1)])
    sess.run(tf.assign(weight[0, 0, d, 0:4], mask))
print(sess.run(weight))
print(weight)
print(weight_s)
print(weight_s1)
print(sess.run(index))
print(tf.reduce_sum(weight_s1, axis=1))

