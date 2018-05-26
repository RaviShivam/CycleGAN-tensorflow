# from matplotlib import pyplot as plt
import os
import zipfile

# from matplotlib import pyplot as plt
import six.moves.urllib as urllib


# import matplotlib as plt


# import matplotlib as plt
# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("object_detection")


def download_zebras2horses_dataset():
    url = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip"
    zip_file = "horse2zebra.zip"
    target_dir = "./datasets/"

    if not os.path.isdir(target_dir + "horse2zebra"):
        print("Downloading dataset...")
        opener = urllib.request.URLopener()
        opener.retrieve(url, zip_file)

        print("Unzipping....")
        zip_ref = zipfile.ZipFile(zip_file, 'r')
        zip_ref.extractall(os.path.join(target_dir))
        zip_ref.close()
        os.remove(zip_file)


def save_bounding_boxes():
    return 0


download_zebras2horses_dataset()
save_bounding_boxes()
