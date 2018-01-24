import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
for variable in tf.global_variables():
    print(variable.name)
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_average_op = ema.apply(tf.global_variables())
for variable in tf.global_variables():
    print(variable.name)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.assign(v, 10))
    sess.run(maintain_average_op)
    saver.save(sess, "model/cpkt/ema.ckpt")
    print(sess.run([v, ema.average(v)]))
