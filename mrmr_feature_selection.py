from skfeature.function.information_theoretical_based.MRMR import mrmr

from load_dataset import load_vfh_data

data, labels, cv_generator = load_vfh_data()

for _, split_idx in cv_generator:
    selected_features, _, _ = mrmr(data[split_idx], labels[split_idx], n_selected_features=304//2)
    print(selected_features)
