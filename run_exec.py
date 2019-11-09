import subprocess
from os import listdir
from os.path import isfile, join

import numpy as np

from utils import old_cats


def get_vhf_representation(file_path):
    args = ("./CPP/build/project", file_path)
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read().decode("utf-8")
    try:
        return eval(output)
    except SyntaxError:
        return None


def folder_with_pcds_to_vhfs(folder_path):
    pcd_filenames = sorted([f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f[-4:] == ".pcd"],
                           key=lambda filename: int(filename.split('.')[0].split('_')[-1]))

    vfh_reps = []
    object_idxs = []
    for filename in pcd_filenames:
        vhf_rep = get_vhf_representation(join(folder_path, filename))
        if vhf_rep:
            vfh_reps.append(vhf_rep)
            object_idxs.append(int(filename.split('.')[0].split('_')[-1]))

    return np.array(vfh_reps), np.array(object_idxs)


# execute only if run as a script
if __name__ == "__main__":
    for i, cat in enumerate(list(reversed(old_cats))):
        folder_path = '/home/gitaar9/AI/COR/Washington_pointclouds/Category/{}_Category'.format(cat)
        category = "_".join(folder_path.split('/')[-1].split('_')[:-1])
        print("{}: Loading category {}".format(i, category))
        try:
            np.load("old_new_dataset/{}.npy".format(category))
            print("skipped")
            continue
        except FileNotFoundError:
            pass
        vfh_reps, object_idxs = folder_with_pcds_to_vhfs(folder_path)
        np.save('/home/gitaar9/AI/COR/CPP_try/old_new_dataset/{}.npy'.format(category), vfh_reps)
        np.save('/home/gitaar9/AI/COR/CPP_try/old_new_dataset/{}_idxs.npy'.format(category), object_idxs)
