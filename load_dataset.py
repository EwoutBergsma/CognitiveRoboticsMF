from utils import cats
from os.path import join
import numpy as np


def load_simple_vfh_dataset(folder_path=None):
    """
    :param folder_path: The path to the folder with .npy files
    :return: x_train, y_train, x_test, y_test
    """
    folder_path = folder_path or "./new_dataset"
    X_train = ()
    Y_train = []
    X_test = ()
    Y_test = []

    for label, cat in enumerate(cats):
        vfh_reps = np.load(join(folder_path, "{}_vfh_reps.npy".format(cat)))
        instance_names = np.load(join(folder_path, "{}_instance_names.npy".format(cat)))

        # Use all but the first instances as training data
        idxs_of_other_instances = np.where(instance_names != instance_names[0])
        train_vfh_reps = vfh_reps[idxs_of_other_instances]
        X_train += (train_vfh_reps,)
        Y_train.extend([label] * train_vfh_reps.shape[0])

        # Use first instance as test data
        idxs_of_first_instance = np.where(instance_names == instance_names[0])
        test_vfh_reps = vfh_reps[idxs_of_first_instance]
        X_test += (test_vfh_reps,)
        Y_test.extend([label] * test_vfh_reps.shape[0])

    return np.concatenate(X_train), np.array(Y_train), np.concatenate(X_test), np.array(Y_test)
