# from matplotlib import pyplot as plt
import os
import zipfile
from os import listdir
from os.path import isfile, join
from segmentation.tf_coco import CocoSegmentation
import pickle

# from matplotlib import pyplot as plt
import six.moves.urllib as urllib

# import matplotlib as plt


# import matplotlib as plt
# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("object_detection")

DATASET_TARGET_DIR = "./datasets/"
DATASET_FOLDER_NAME = "horse2zebra"
TEST_A = DATASET_TARGET_DIR + DATASET_FOLDER_NAME + "/testA"


def download_zebras2horses_dataset():
    url = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip"
    extension = ".zip"

    if not os.path.isdir(DATASET_TARGET_DIR + DATASET_FOLDER_NAME):
        print("Downloading dataset...")
        opener = urllib.request.URLopener()
        opener.retrieve(url, DATASET_FOLDER_NAME + extension)

        print("Unzipping....")
        zip_ref = zipfile.ZipFile(DATASET_FOLDER_NAME, 'r')
        zip_ref.extractall(os.path.join(DATASET_TARGET_DIR))
        zip_ref.close()
        os.remove(DATASET_FOLDER_NAME + extension)


def get_files_from_dataset_folder():
    paths = []
    paths.append(DATASET_TARGET_DIR + DATASET_FOLDER_NAME + "/testA")
    paths.append(DATASET_TARGET_DIR + DATASET_FOLDER_NAME + "/testB")
    paths.append(DATASET_TARGET_DIR + DATASET_FOLDER_NAME + "/trainA")
    paths.append(DATASET_TARGET_DIR + DATASET_FOLDER_NAME + "/trainB")

    files = []
    for path in paths:
        files_in_subfolder = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        files.extend(files_in_subfolder)

    return files


def save_mask(model, files):

    masks_dict = model.segment_objects(files)
    return masks_dict


def save_as_pickle(masks):
    with open('masks.pickle', 'wb') as handle:
        pickle.dump(masks, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    download_zebras2horses_dataset()
    files = get_files_from_dataset_folder()
    coco_segmentation = CocoSegmentation()
    masks = save_mask(coco_segmentation, files)
    save_as_pickle(masks)
