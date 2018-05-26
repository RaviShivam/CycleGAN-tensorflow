import tensorflow as tf
import numpy as np

b = tf.constant([[1, 2, 3], [1, 2, 3]])
a = tf.constant([1, 1, 1])
a = tf.expand_dims(a, 0)
a = tf.pad(a, [[0, 1], [0, 0]], "CONSTANT")
c = a + b
with tf.Session() as sess:
    print(sess.run(c))
