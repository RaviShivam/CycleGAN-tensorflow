import tensorflow as tf
import numpy as np

up = tf.constant(list(np.ones(4).astype(np.int32)), dtype=tf.int32)
bb = tf.constant([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], tf.int32)
loops = tf.shape(bb)[0]

cond = lambda i, _: tf.less(i, loops)
b = lambda i, bb: tf.add(up, bb[i])

init = (0, bb)
r = tf.while_loop(cond, b, init)

with tf.Session() as sess:
    print(sess.run(r))
