from utils import *
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from segmentation.DeeplabModel import vis_segmentation
import tensorflow as tf
import pickle

# bbs = load_bounding_boxes_complete("horse2zebra")
# imgs = ["datasets/horse2zebra/trainA/000000000049.jpg", "datasets/horse2zebra/trainB/000000051576.jpg"]
#
# a, b = load_bounding_box_real(imgs, bbs[0], bbs[1])
# print(a, b)
with open('./datasets/horse2zebra/bboxA.p', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    p = u.load()
img = imread("datasets/horse2zebra/trainA/000000011818.jpg")
mask = create_bbox_mask(p["000000011818.jpg"].astype(int), img.shape)
print(img.shape, mask.shape)
plt.imshow(mask.astype(np.uint8))
plt.show()

