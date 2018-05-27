import tensorflow as tf
from module import generator_resnet
import pickle
import numpy as np
from PIL import Image

f = open("datasets/horse2zebra/maskA.p", mode="rb")
data = pickle.load(f)
m = np.array(data["n02381460_20.jpg"], dtype=np.uint8)
real_A = np.array(Image.open("datasets/horse2zebra/testA/n02381460_20.jpg"))
real_A = np.expand_dims(real_A, 0)

print(real_A.dtype)
print(m.dtype)

gen = generator_resnet(image=real_A, mask=m, options=None,
                       reuse=False, name="generatorA2B")
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    g = sess.run(gen)
    g = np.array(g[0, :, :, :], dtype=np.uint8)
    print(g.shape)
    im = Image.fromarray(g)
    im.show()
    Image.open("datasets/horse2zebra/testA/n02381460_20.jpg").show()
# a = tf.zeros([100, 100])
# b = tf.stack([a, a, a], axis=2)
# b = tf.expand_dims(b, 0)
#
# with tf.Session() as sess:
#     g = sess.run(b)
#     print(g)
#     print(g.shape)
