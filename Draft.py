import tensorflow as tf
import numpy as np
sess = tf.Session()
list = []
weight1 = tf.get_variable("6", [1], initializer=tf.constant_initializer(0))
init = tf.global_variables_initializer()
one = np.array([1])
two = np.array([2])
sess.run(init)
print(sess.run(weight1))
sess.run(tf.assign(weight1, one))
print(sess.run(weight1))
sess.run(tf.assign(weight1, two))
print(sess.run(weight1))



