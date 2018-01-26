import tensorflow as tf

A = tf.get_variable('A', [2, 2], initializer=tf.constant_initializer(0.0))
B = tf.get_variable('B', [2, 2], initializer=tf.constant_initializer(0.0))
tf.add_to_collection('H', A)
tf.add_to_collection('H', B)

a = tf.get_variable('a', [2, 2], initializer=tf.constant_initializer(0.0))
b = tf.get_variable('b', [2, 2], initializer=tf.constant_initializer(0.0))
    
tf.add_to_collection('H', a)
tf.add_to_collection('H', b)

print(tf.get_collection('H'))