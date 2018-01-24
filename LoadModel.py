import tensorflow as tf
saver = tf.train.import_meta_graph("model/cpkt/test_model.meta")
with tf.Session() as sess:
    saver.restore(sess, "model/cpkt/test_model")
    print("%d ops in the final graph" % len(tf.get_default_graph().as_graph_def().node))