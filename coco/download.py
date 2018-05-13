# Check if coco is installed
from importlib import util
import sys
coco_loader = util.find_spec('pycocotools')
if (coco_loader is None):
    sys.exit('"pycocotools" not installed. Run "sudo make install" in datasets/coco/coco-api/PythonAPI (linux only), but modify the Makefile there to use "python3" instead of "python" if that is necessary to run python3 on your system')

import argparse
import os
import zipfile
from urllib.request import urlretrieve, urlopen
from PIL import Image
from io import BytesIO

from pycocotools.coco import COCO
import pycocotools.mask

parser = argparse.ArgumentParser()

parser.add_argument("-ds", "--dataset", default="train", help="Dataset [train, val]")
parser.add_argument("-rs", "--resolution", default=512, help="Resolution of downloaded images", type=int)
parser.add_argument("-c", "--category", default="horse", help="Which COCO category to download")

args = parser.parse_args()

print('Attempting to download {} data for "{}" in resolution {}. One eye can of patience please.'.format(args.dataset, args.category, args.resolution))
print('Options: e.g. "--dataset val", "--resolution 420", "--category zebra"')

dir_path = os.path.dirname(os.path.realpath(__file__))

def load_annotations():
    annotation_filename = 'instances_{}2017.json'.format(args.dataset)
    annotation_dir = os.path.join(dir_path, 'coco-api', 'annotations')
    annotation_path = os.path.join(annotation_dir, annotation_filename)

    if not os.path.isfile(annotation_path):
        os.makedirs(annotation_dir)
        
        url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        download_path = os.path.join( os.path.join(dir_path, 'coco-api'), 'annotations_trainval2017.zip')

        print('Annotations not found. Downloading now, this might take a while...')
        urlretrieve(url, download_path)

        print('unzipping...')
        zip_ref = zipfile.ZipFile(download_path, 'r')
        zip_ref.extractall(annotation_dir)
        zip_ref.close()

        print('cleaning up...')
        os.remove(download_path)

    return COCO(annotation_path)


def getFileName(url):
    return url[url.rfind("/")+1:]


def saveBoundingBoxes(coco, imgs):
    bboxes = {}
    for img in imgs:
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        w, h = img['width'], img['height']
        # Store all bounding boxes
        anns = coco.loadAnns([annIds[0]])
        img_masks = [coco.annToMask(ann) for ann in anns]
        bin_masks = [pycocotools.mask.encode(img_mask) for img_mask in img_masks]
        img_bboxes = [pycocotools.mask.toBbox(bin_mask) for bin_mask in bin_masks]
        sqr_bboxes = [adjust_bbox_square(bbox, (w, h), args.resolution) for bbox in img_bboxes]
        filename = getFileName(img['coco_url'])
        bboxes[filename] = sqr_bboxes
    
    import pickle

    bbox_path = os.path.normpath(os.path.join(dir_path, '..', 'datasets', '{}_{}_{}_bounding-boxes.p'.format(args.category, args.resolution, args.dataset)))
    with open(bbox_path, 'wb') as fp:
        pickle.dump(bboxes, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved bboxes as pickle into "{}"'.format(bbox_path))


def adjust_bbox_square(bbox, im_size, max_size):
    x, y = im_size
    max_s = max(x, y)
    new_size = int(x / max_s * max_size), int(y / max_s * max_size)

    scalar = max_size / max_s
    t_x, t_y = (max_size - new_size[0]) // 2, (max_size - new_size[1]) // 2 # translation
    res = [bbox[0] * scalar + t_x, bbox[1] * scalar + t_y, bbox[2] * scalar, bbox[3] * scalar]
    return [round(x) for x in res]


def make_square(im, max_size=256, fill_color=(0, 0, 0)):
    x, y = im.size
    max_s = max(x, y)
    new_size = int(x / max_s * max_size), int(y / max_s * max_size)
    paste_pos = (max_size - new_size[0]) // 2, (max_size - new_size[1]) // 2
    im = im.resize(new_size, resample=Image.BILINEAR)

    new_im = Image.new('RGB', (max_size, max_size), fill_color)
    new_im.paste(im, paste_pos)
    return new_im


def downloadImages(imgs):
    # Download images in parallel
    from multiprocessing.dummy import Pool

    download_path = os.path.normpath(os.path.join(dir_path, '..', 'datasets', '{}_{}_{}'.format(args.category, args.resolution, args.dataset)))
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    def download_img(url):
        filename = getFileName(url)
        path = os.path.join(download_path, filename)

        f = urlopen(url)
        jpeg_str = f.read()
        original_im = Image.open(BytesIO(jpeg_str))
        square_im = make_square(original_im, args.resolution)
        square_im.save(path)

    # download 4 files at a time
    urls = [img['coco_url'] for img in imgs]
    Pool(4).map(download_img, urls) 


if __name__ == '__main__':
    coco = load_annotations()
    category_to_download = args.category

    catIds = coco.getCatIds(catNms=[category_to_download])
    imgIds = coco.getImgIds(catIds=catIds)
    imgs = coco.loadImgs(imgIds)
    print('Found {} images of the "{}" category'.format(len(imgIds), category_to_download))

    data_path = os.path.join(dir_path, 'data')
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    print('Saving bounding boxes...')
    saveBoundingBoxes(coco, imgs)

    print('Downloading images (might take a while...)')
    downloadImages(imgs)

    print('Done!')
