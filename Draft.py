import tensorflow as tf

_input = tf.placeholder(tf.float32, [None, 32, 32, 32], 'images')
splited_features = tf.split(_input, num_or_size_splits=4, axis=3)
for features in splited_features:
    
