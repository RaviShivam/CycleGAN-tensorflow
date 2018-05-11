from utils import *
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf

a = tf.ones([3, 3])
paddings = tf.constant([[4,1], [2,2]])
b = tf.pad(a, paddings)

c = tf.shape(b)

x, y = [c[0], c[1]]

with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(x))

