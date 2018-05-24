"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
import tensorflow as tf
import copy
import pickle

try:
    _imread = scipy.misc.imread
except AttributeError:
    from imageio import imread as _imread

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand() * self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand() * self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def load_bounding_boxes_complete(dataset):
    with open('./datasets/{}/bboxA.p'.format(dataset), 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        bboxAfull = u.load()

    with open('./datasets/{}/bboxB.p'.format(dataset), 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        bboxBfull = u.load()

    return [bboxAfull, bboxBfull]


def load_bounding_box_real(batch_files, bboxAfull, bboxBfull):
    bboxA_real = np.array(bboxAfull[batch_files[0].split("/")[-1]])
    bboxB_real = np.array(bboxBfull[batch_files[1].split("/")[-1]])
    return bboxA_real.astype(int), bboxB_real.astype(int)



def load_test_data(image_path, image_size):
    img = imread(image_path)
    img = scipy.misc.imresize(img, [image_size, image_size])
    img = img / 127.5 - 1
    return img


def load_train_data(image_path, fine_size):
    img_A = imread(image_path[0])
    img_B = imread(image_path[1])
    img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
    img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])

    img_A = img_A / 127.5 - 1.
    img_B = img_B / 127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def create_bbox_mask(bbox, full_image_shape):
    y0, x0, w, h = bbox
    nbbx = [x0, y0, h, w]
    mask = np.ones([h, w])
    a = [x0, full_image_shape[0] - (x0 + h)]
    b = [y0, full_image_shape[1] - (y0 + w)]
    padding = [a, b]
    mask = np.pad(mask, padding, mode="constant")
    return mask
# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale=False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, is_grayscale=False):
    if (is_grayscale):
        return _imread(path, flatten=True).astype(np.float)
    else:
        return _imread(path, mode='RGB').astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(
        x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.

