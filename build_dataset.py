from os import listdir
from os.path import isfile, join, isdir

import numpy as np
from imageio import imread
from skimage.transform import resize

from run_exec import get_vhf_representation
from utils import cats


# from matplotlib import pyplot as plt
EVAL_DATASET_PATH = "/home/gitaar9/AI/COR/CPP_try/raw_datasets/rgbd-dataset_eval"
PC_DATASET_PATH = "/home/gitaar9/Downloads/finished_tars_rgbd/rgbd-dataset"
OUTPUT_DATASET_PATH = "/home/gitaar9/AI/COR/CPP_try/new_dataset"


def read_object(category_name, object_name):
    object_path = join(EVAL_DATASET_PATH, category_name, object_name)
    # Get the filenames from the txt files
    filenames = [f[:-8] for f in listdir(object_path)
                 if isfile(join(object_path, f)) and f[-8:] == "_loc.txt"]

    resized_images = []
    vfh_representations = []

    print("{} contains {} angles".format(object_name, len(filenames)))
    for filename in filenames:
        # Calculate the vhf representation
        point_cloud_path = join(PC_DATASET_PATH, category_name, object_name, "{}.pcd".format(filename))
        vhf_rep = get_vhf_representation(point_cloud_path)
        if not vhf_rep:
            continue
        vfh_representations.append(vhf_rep)
        # Load the image
        im = imread(join(object_path, '{}_crop.png'.format(filename)))
        resized = resize(im, (224, 224))
        resized_images.append(resized)

    return np.array(resized_images), np.array(vfh_representations)


def read_category(category_name):
    category_path = join(EVAL_DATASET_PATH, category_name)
    objects_names = [d for d in listdir(category_path) if isdir(join(category_path, d))]
    print("Found {} instances of {}".format(len(objects_names), category_name))

    object_images = ()
    object_vhf_representations = ()
    object_aggregated_names = ()

    for object_name in objects_names:
        resized_images, vhf_representations = read_object(category_name, object_name)
        object_images += (resized_images,)
        object_vhf_representations += (vhf_representations,)
        object_aggregated_names += ([object_name] * resized_images.shape[0],)

    return np.concatenate(object_images), np.concatenate(object_vhf_representations), \
           np.concatenate(object_aggregated_names)


# execute only if run as a script
if __name__ == "__main__":

    category_names = [category_name for category_name in cats]
    for category_name in category_names:
        # Check if file already exist
        try:
            np.load(join(OUTPUT_DATASET_PATH, "{}_instance_names.npy".format(category_name)))
            print("SKIPPED: npy files for {} already exist".format(category_name))
            continue
        except FileNotFoundError:
            pass

        print("Creating npy files for {}".format(category_name))
        # Load the data in np arrays files
        category_image_data, category_vhf_data, category_name_data = read_category(category_name)
        print("Total sizes for ", category_name, ": ", category_image_data.shape, category_vhf_data.shape,
              category_name_data.shape)

        # Save the np arrays to npy files
        np.save(join(OUTPUT_DATASET_PATH, "{}_smoothed_images.npy".format(category_name)), category_image_data)
        np.save(join(OUTPUT_DATASET_PATH, "{}_vfh_reps.npy".format(category_name)), category_vhf_data)
        np.save(join(OUTPUT_DATASET_PATH, "{}_instance_names.npy".format(category_name)), category_name_data)
