from utils import *
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf

im = np.array(Image.open("test.jpg"))
imtf = tf.constant(im, dtype=tf.float32)
bbox = tf.constant([140, 0, 50, 100, 100, 100], dtype=tf.int32)

newim = tf.py_func(crop_and_resize_real, [imtf, bbox], tf.float32)
newim = tf.zeros_like(newim)
finalim = tf.py_func(resize_and_refit_fake, [imtf, newim, bbox], tf.float32)


var = np.random.random((1, 10, 10, 10))
a = var[0, :, :, :]
b = np.reshape(a, (1, 10, 10, 10))
# with tf.Session() as sess:
#     plt.imshow(sess.run(finalim))
#     plt.show()
