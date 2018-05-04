import os
import os.path
from io import BytesIO
from six.moves import urllib
from PIL import Image
import time

from DeeplabModel import DeepLabModel, vis_segmentation

# FROM https://github.com/tensorflow/models/tree/master/research/deeplab

# Enable this to run on CPU instead of GPU (if GPU is used by default)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def load_model(MODEL_NAME='mobilenetv2_coco_voctrainaug'):

  ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
  _TARBALL_NAME = '{}.tar.gz'.format(MODEL_NAME)
  _TARBALL_PATH = os.path.join(ROOT_DIR, 'models', _TARBALL_NAME)

  if not os.path.exists(_TARBALL_PATH):
    print('model file not found. Downloading now, this might take a while...')

    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    _MODEL_URLS = {
      'mobilenetv2_coco_voctrainaug': 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
      'mobilenetv2_coco_voctrainval': 'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
      'xception_coco_voctrainaug': 'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
      'xception_coco_voctrainval': 'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    }

    urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], _TARBALL_PATH)
    print('download completed!')

  print('loading DeepLab model...')
  MODEL = DeepLabModel(_TARBALL_PATH)
  print('model loaded successfully!')
  return MODEL


if __name__ == '__main__':
  # Demonstration: It downloads an image and applies the segmentation on it

  MODEL = load_model()
  url = ('https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/img/image1.jpg?raw=true')

  try:
    print('downloading sample image...')
    f = urllib.request.urlopen(url)
    jpeg_str = f.read()
    orignal_im = Image.open(BytesIO(jpeg_str))

    t0 = time.clock()

    print('running model...')
    resized_im, seg_map = MODEL.run(orignal_im)

    t1 = time.clock()
    total = t1-t0

    print('Successfully segmented the sample image. Time: {} seconds'.format(total))

    vis_segmentation(resized_im, seg_map)
  except IOError:
    print('Cannot retrieve image. Please check url: ' + url)
